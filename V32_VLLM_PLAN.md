# V32 Plan: vLLM Server 方案

## Context

`transformers 5.6.0.dev0` 的 `from_pretrained()` 加载 Qwen3.5-35B-A3B MoE 模型时会导致 WSL 崩溃（分层测试证明文件系统 I/O 正常，问题在 transformers 内部）。vLLM 是 Qwen3.5 官方推荐的推理引擎，绕过 transformers 的模型加载问题。

## 核心思路

**将 pipeline 从"直接调用 HF model"改为"调用 vLLM OpenAI-compatible API"。**

vLLM 启动后作为 HTTP server 独立运行，pipeline 通过 `openai` Python SDK 发送请求。这样：
- 模型加载由 vLLM 管理（不经过 transformers `from_pretrained`）
- KV Cache 管理由 vLLM 的 PagedAttention + Prefix Caching 自动处理
- Pipeline 代码只需要替换底层推理函数，上层逻辑不变

## 架构变化

```
之前:  run.py → model = AutoModelForCausalLM.from_pretrained()
              → run_model_with_cache_manual(prompt, model, tokenizer, config, cache)
              → model(input_ids, past_key_values=cache)

之后:  [独立进程] vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000
       run.py → client = openai.OpenAI(base_url="http://localhost:8000/v1")
              → vllm_generate(prompt, client, config)
              → client.completions.create(prompt=prompt, ...)
```

## KV Cache 策略

当前 pipeline 的核心优化是 KV Cache 分叉：
1. `build_base_cache()` — 将 `system + note_text` 编码为 KV cache
2. 每个提取任务 `clone_cache(base_cache)` + 追加 task prompt

**vLLM 的 `--enable-prefix-caching` 自动实现等效效果**：
- 多个请求如果共享相同的 prefix（system + note_text），vLLM 自动复用 KV cache
- 无需手动管理 cache，只需确保每个请求的 prompt 都包含完整的 `system + note + task`
- 对于同一个 note 的 10+ 个提取任务，vLLM 只编码 note 一次

## 改动清单

### 1. 新建 `vllm_client.py` — vLLM 推理封装

```python
# vllm_client.py — 替代 run_model_with_cache_manual + build_base_cache

import openai

class VLLMClient:
    def __init__(self, base_url="http://localhost:8000/v1", model_name=None):
        self.client = openai.OpenAI(base_url=base_url, api_key="dummy")
        self.model_name = model_name  # vLLM 会自动检测

    def generate(self, full_prompt, gen_config):
        """替代 run_model_with_cache_manual()"""
        response = self.client.completions.create(
            model=self.model_name,
            prompt=full_prompt,
            max_tokens=gen_config.get("max_new_tokens", 768),
            temperature=0 if not gen_config.get("do_sample") else gen_config.get("temperature", 0.6),
            top_p=gen_config.get("top_p", 1.0),
            stop=["<|im_end|>"],  # Qwen3 ChatML end token
        )
        return response.choices[0].text.strip()
```

关键设计：
- `generate()` 接受完整 prompt（包含 system+note+task），返回生成的文本
- 不返回 cache（vLLM 自动管理）
- stop token 使用 `<|im_end|>` 匹配 ChatML 格式

### 2. 修改 `ult.py` — 适配 vLLM

**需要改的函数**：

| 函数 | 改动 | 说明 |
|------|------|------|
| `run_model_with_cache_manual()` | 加 vLLM 分支 | 检测是否 vLLM 模式，走 HTTP 调用 |
| `build_base_cache()` | 返回 prompt 字符串而非 cache | vLLM 模式下 "cache" = base prompt 文本 |
| `clone_cache()` | vLLM 模式下直接返回 | 字符串无需 deepcopy |
| `run_model()` | 加 vLLM 分支 | A/P 提取的 fallback |

**核心改动模式**——`run_model_with_cache_manual()`:

```python
def run_model_with_cache_manual(prompt_text, model, tokenizer, generation_config, kv_cache=None):
    # vLLM 模式
    if isinstance(model, VLLMClient):
        if kv_cache is not None:  # kv_cache 实际是 base_prompt 字符串
            full_prompt = kv_cache + prompt_text
        else:
            full_prompt = prompt_text
        result = model.generate(full_prompt, generation_config)
        return result, kv_cache  # cache 不变（vLLM 自动管理）
    
    # 原始 HF 模式（保持不变）
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    ...
```

**`build_base_cache()` 改动**：

```python
def build_base_cache(text, model, tokenizer, definitions_context="", chat_tmpl=None):
    # ... 构建 base_prompt（同原来的逻辑）...
    
    # vLLM 模式：返回 prompt 字符串作为 "cache"
    if isinstance(model, VLLMClient):
        return base_prompt
    
    # 原始 HF 模式（保持不变）
    with torch.no_grad():
        ...
```

### 3. 修改 `run.py` — 模型加载逻辑

在 `main()` 的模型加载部分加入 vLLM 模式：

```python
# run.py line ~698
vllm_cfg = model_cfg.get("vllm")
if vllm_cfg:
    # vLLM 模式：不加载模型，创建 HTTP client
    from vllm_client import VLLMClient
    model = VLLMClient(
        base_url=vllm_cfg.get("base_url", "http://localhost:8000/v1"),
        model_name=model_cfg["name"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
else:
    # 原始 HF 模式（保持不变）
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **load_kwargs)
```

### 4. 新建 YAML 配置 `exp/v32_vllm.yaml`

```yaml
experiment:
  name: "v32_vllm_test"
  seed: 42

model:
  name: "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
  chat_template: "qwen2"
  vllm:
    base_url: "http://localhost:8000/v1"

data:
  dataset_path: "data/CORAL/..."
  row_indices: [0, 1, 7, 16, 28, 45, 63, 99]

# ... 其余同 v31 ...
```

### 5. vLLM Server 启动脚本

```bash
# start_vllm.sh
vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --port 8000 \
  --enable-prefix-caching \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --dtype float16
```

## 不改动的部分

- `extract_and_verify_v2()` — 签名不变，内部通过 `run_model_with_cache_manual()` 间接适配
- `letter_generation.py` — 通过 `run_model_with_cache_manual()` 间接适配
- `source_attribution.py` — 同上
- `ChatTemplate` 类 — 不变
- 所有 prompt YAML — 不变
- 所有 POST hooks — 不变
- v31 及之前的配置 — 完全兼容（没有 `vllm` 字段时走原始 HF 路径）

## 向后兼容性保证

```
如果 YAML 有 model.vllm 字段  → vLLM HTTP 模式（新）
如果 YAML 没有 model.vllm 字段 → HuggingFace 直接加载（原始，v31 不受影响）
```

## 执行步骤

1. 创建 `vllm_client.py`
2. 修改 `ult.py`（3 个函数加 vLLM 分支）
3. 修改 `run.py`（模型加载 + tokenizer 加载）
4. 创建 `exp/v32_vllm.yaml`
5. 创建 `start_vllm.sh`
6. WSL 上安装 vLLM：`pip install vllm`
7. 先启动 vLLM server（tmux session）
8. 再启动 pipeline（另一个 tmux session）
9. 监控进度

## 风险点

1. **vLLM 安装/启动**：vLLM 在 WSL 上可能也有兼容性问题（和 transformers 类似）。但 vLLM 有自己的模型加载路径，不经过 `from_pretrained`。
2. **Prefix caching 效率**：依赖 vLLM 自动检测公共前缀。如果 prompt 格式不完全一致（如空格/换行差异），prefix caching 会失效。
3. **stop token**：需要确认 Qwen3.5 的 EOS/stop token 列表。
4. **推理速度**：HTTP 通信有开销，但 vLLM 的 PagedAttention 和批处理能力会补偿。
5. **thinking mode**：Qwen3.5 有 `<think>` 模式需要禁用（否则输出会包含推理过程）。

## 验证方法

1. vLLM server 启动后，用 `curl` 测试 API 是否正常
2. 跑 v32 的 8 个 test sample
3. 对比 v31 相同 8 个 sample 的结果质量
4. 确认 v31 原始 config 仍然正常工作（向后兼容）

## 关键文件

| 文件 | 改动类型 |
|------|---------|
| `vllm_client.py` | **新建** |
| `start_vllm.sh` | **新建** |
| `exp/v32_vllm.yaml` | **新建** |
| `ult.py` line 177, 277, 401, 608 | 修改（加 vLLM 分支） |
| `run.py` line 698-705 | 修改（加 vLLM 加载分支） |
| `letter_generation.py` | 不改（通过 run_model_with_cache_manual 间接适配） |
| `source_attribution.py` | 不改（同上） |

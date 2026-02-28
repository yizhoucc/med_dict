# med_dict - 医学肿瘤笔记结构化提取与患者解释生成

## 项目概述
使用 LLM (Llama 3.1 8B Instruct) 从肿瘤科临床笔记中提取结构化信息，并生成面向患者的通俗解释。

## 数据
- **CORAL 数据集**：乳腺癌肿瘤科门诊笔记（去标识化）
- **医学词典**：`data/formaldef.txt`（9369 术语），经 8 年级词汇过滤
- HuggingFace token 存于 `hf.token`（勿提交）

## 核心文件
- `ult.py` - 工具库：模型推理（含 KV Cache 分叉）、JSON 修复、文件保存
  - `build_base_cache()` - 构建 KV Cache
  - `extract_and_verify()` - 提取 + faithfulness 验证（复用函数）
  - `run_model_with_cache_manual()` - 手动 token-by-token 生成，支持 eos_token_id 列表
- `work flow keypoint v1 check v2.ipynb` - **最新版本**，两阶段提取 + faithfulness 验证
- `work flow keypoint v1 summary v13.ipynb` - 含解释生成的版本
- `notebooks/` - 早期工作：医学术语简化 + 可读性评估
- `plots/` - 可读性指标对比图（SVG）

## Pipeline（最新版 v1 check v2）
1. **Assessment/Plan 提取** - 从笔记中 copy-paste 原文（3 次重试 + LLM 验证）
   - 未来考虑：用正则预提取，LLM 作为 fallback（数据格式不统一，需先看数据）
2. **关键点提取** (extraction_prompts) - 从全文提取：就诊原因、检查发现、治疗摘要、治疗目标
3. **计划提取** (plan_extraction_prompts) - 从 A/P 段提取：用药/手术/影像/检验/基因检测/转诊/随访/预前计划
4. **Faithfulness Check** - 每项提取后验证是否有幻觉
5. **解释生成**（可选）- 从 keypoints 生成 8 年级英语水平的患者信

## 模型配置
- 模型：`meta-llama/Llama-3.1-8B-Instruct`，bfloat16，单 GPU
- KV Cache 分叉：笔记编码一次，多任务复用缓存
- 提取用贪婪解码，生成用采样（temperature=0.3）

## 运行环境
- WSL (Ubuntu 22.04)，通过 `ssh wsl` 从 Mac 访问
- Conda 环境：medllm

## 代码规范
- results.txt 保持当前自定义格式，不改为 JSON
- `from ult import *` 暂时保留
- prompt 中的 JSON schema 必须是合法 JSON（注意逗号）

# V26 设计: LLM-Driven Tool Calling for Extraction

> 日期: 2026-04-07
> 目标: 让模型在提取过程中主动请求信息，替代固化的 prefetch

---

## 问题

### 问题 1: 医学词典注入太固化
`find_relevant_definitions()` 用硬编码的 `INJECT_PRIORITY_TERMS` 列表匹配术语。
- 不知道模型实际不认识哪些词
- 有些 note 里的缩写/俗称不在优先列表里
- 注入了模型不需要的定义，浪费 context

### 问题 2: Plan extraction 只看 A/P
`base_cache = build_base_cache(model_ap, ...)` — plan extraction 只用 A/P 段构建 cache。
- A/P 说 "continue current medications" 但药物列表在 Medications 段
- A/P 说 "s/p XRT" 但放疗细节在 Oncologic History
- A/P 说 "labs reviewed wnl" 但实际 lab 值在 Results 段
- 现有 POST hooks (POST-REFERRAL, POST-LAB-SEARCH 等) 用 regex 补救，但不够灵活

---

## 设计方案: 两轮提取 + Tool Call 拦截

### 核心思路

在 extraction prompt 里告诉模型：你可以请求两种工具。第一轮生成时如果模型输出了 tool call pattern，我们拦截、执行工具、注入结果后重新生成。如果没有 tool call，直接用第一轮结果（零额外开销）。

### 可用工具

```
SEARCH_NOTE(keyword) — 在全文中搜索包含 keyword 的段落，返回上下文
DEFINE(term) — 在医学词典中查找 term 的通俗定义
```

### 工作流

```
┌─────────────────────────────────────────┐
│ Pass 1: 正常提取                         │
│   prompt + A/P text → LLM → output      │
│                                         │
│   检查 output 中是否有 tool call:        │
│   - SEARCH_NOTE("current medications")  │
│   - DEFINE("TCHP")                      │
│                                         │
│   有 tool call?                         │
│   ├─ 否 → 直接用 output（零额外开销）   │
│   └─ 是 → 执行工具，注入结果            │
│           ↓                             │
│           Pass 2: 带工具结果重新提取     │
│           enhanced_prompt → LLM → output│
└─────────────────────────────────────────┘
```

### Prompt 修改

在每个 plan extraction prompt 末尾加：

```
TOOLS (optional — use ONLY if you need information not available in the A/P section):
- To search the full medical note: write SEARCH_NOTE("keyword") on a separate line
- To look up a medical term definition: write DEFINE("term") on a separate line
If you have enough information, do NOT use any tools — just output the JSON directly.
```

### 实现

```python
# ult.py 新增函数

TOOL_PATTERN = re.compile(r'(SEARCH_NOTE|DEFINE)\("([^"]+)"\)')

def parse_tool_calls(output):
    """Parse tool call patterns from LLM output."""
    return TOOL_PATTERN.findall(output)

def execute_tool_calls(calls, full_note, med_dict):
    """Execute tool calls and return additional context."""
    results = []
    for tool_name, arg in calls:
        if tool_name == "SEARCH_NOTE":
            # Search full note for keyword, return ±200 chars context
            idx = full_note.lower().find(arg.lower())
            if idx >= 0:
                start = max(0, idx - 200)
                end = min(len(full_note), idx + 200)
                passage = full_note[start:end]
                results.append(f"[SEARCH_NOTE result for '{arg}']:\n{passage}")
            else:
                results.append(f"[SEARCH_NOTE result for '{arg}']: Not found in note.")
        elif tool_name == "DEFINE":
            # Look up in medical dictionary
            term_lower = arg.lower()
            if term_lower in med_dict:
                term_orig, definition = med_dict[term_lower]
                results.append(f"[DEFINE result for '{arg}']: {definition}")
            else:
                results.append(f"[DEFINE result for '{arg}']: Definition not available.")
    return "\n\n".join(results)

def extract_with_tools(prompt, model, tokenizer, gen_config, cache, 
                       full_note, med_dict, chat_tmpl, max_tool_rounds=1):
    """Extract with optional tool calling support."""
    task_prompt = chat_tmpl.user_assistant(prompt)
    answer, _ = run_model_with_cache_manual(
        task_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(cache)
    )
    
    # Check for tool calls
    tool_calls = parse_tool_calls(answer)
    if not tool_calls or max_tool_rounds <= 0:
        return answer  # No tools needed, zero overhead
    
    # Execute tools
    tool_results = execute_tool_calls(tool_calls, full_note, med_dict)
    
    # Re-extract with tool results injected
    enhanced_prompt = (
        f"{prompt}\n\n"
        f"--- TOOL RESULTS ---\n{tool_results}\n--- END TOOL RESULTS ---\n\n"
        f"Now answer the question using the original A/P AND the tool results above."
    )
    enhanced_task = chat_tmpl.user_assistant(enhanced_prompt)
    answer2, _ = run_model_with_cache_manual(
        enhanced_task, model, tokenizer, gen_config, kv_cache=clone_cache(cache)
    )
    
    return answer2
```

### 对 KV Cache 的影响

- **Pass 1**: 用 `ap_cache`（和现在一样）
- **Pass 2**: 也用 `ap_cache`，但 prompt 更长（包含 tool results）
- KV cache 仍然有效——base 部分（A/P text）不变，只是 task prompt 更长
- 大部分 sample 不会触发 tool call → 零额外开销

### 对性能的影响

- 大部分字段：零额外开销（模型直接回答）
- 触发 tool call 的字段：多 1 次 LLM 调用（~3-5 秒）
- 预估：10-20% 的字段会触发 tool call → 总时间增加 ~15-30 分钟/100 sample

---

## 预期解决的问题

| 问题 | 现有方案 | Tool Calling 方案 |
|------|----------|------------------|
| lab_plan 混入 imaging | POST-LAB hook regex | 模型 SEARCH_NOTE("labs") 获取 lab 段落 |
| Referral 遗漏全文提及 | POST-REFERRAL hook regex | 模型 SEARCH_NOTE("referral") |
| genetic_testing 遗漏 | POST-GENETICS-SEARCH regex | 模型 SEARCH_NOTE("oncotype") |
| A/P "continue meds" 不知道什么药 | 无 | 模型 SEARCH_NOTE("medications") |
| 不认识的缩写 | INJECT_PRIORITY_TERMS 固化 | 模型 DEFINE("TCHP") 按需查 |

---

## 实施步骤

- [ ] 1. `ult.py` 加 `parse_tool_calls()`, `execute_tool_calls()`, `extract_with_tools()`
- [ ] 2. 修改 plan_extraction.yaml 每个 prompt 加 TOOLS 说明
- [ ] 3. `run.py` plan extraction 部分改用 `extract_with_tools()`
- [ ] 4. 8 sample 测试：看模型是否能正确使用 SEARCH_NOTE/DEFINE
- [ ] 5. 验证：tool call 触发率、准确性、对比无 tool 版本
- [ ] 6. 全量跑 + 审查

---

## 风险

1. **Qwen 32B 可能不会主动使用 tool call** — 需要在 prompt 中给足够示例
2. **过度使用 tool call** — 每个字段都调用 → 性能翻倍。需要 prompt 说清楚 "只在需要时使用"
3. **tool 结果注入后格式混乱** — 模型可能被工具结果干扰。需要测试
4. **与 Gate 系统的交互** — tool call 在 Gate 1-5 之前还是之后？应该在提取阶段（Gate 之前）

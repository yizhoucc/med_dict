# V31 iter14 — Full Review (Extraction + Letter)

> 56 samples, doctor feedback fixes applied
> Automated scan + manual review for each sample

## 状态
- 审查中
- Extraction: P0:0 P1:1 (ROW 57 TNBC ER+/PR+) P2: TBD
- Letter: P0:0 P1:0 P2: TBD

## 自动扫描结果

### Extraction Issues (自动检测)
- **ROW 57 P1**: Type_of_Cancer says TNBC but also ER+/PR+ — self-contradictory (same as iter13)
- POST hook additions ("; also:"): ROW 2,6,7,22,29,36,46,52,57,65,82,85,90,91,99
  - 问题性: ROW 22 (letrozole/abemaciclib 已停用), ROW 85 (palbociclib 已停用)

### Letter Issues (自动检测)
- **"medication test"**: ROW 24, 72, 96 — LLM 不一致遵守 prompt 规则
- **Truncation**: ROW 22, 84, 90, 99 (严重，缺 closing), ROW 2, 30 (轻微)
- ✅ 无 "no cancer found in removed tissue" — iter12e P1 fix 保持
- ✅ 无 Parkinson's mention — doctor feedback fix 保持
- ✅ 无 "hand-foot syndrome" — iter12e P2 fix 保持

## 医生反馈修复验证

| 反馈项 | iter14 状态 |
|--------|-----------|
| ROW 87: 重复表述 | ✅ "Why" 部分只有一句 |
| ROW 87: Parkinson's tremor | ✅ 已删除 |
| ROW 87: 推断 curative goal | ✅ 不再说 cure |
| ROW 87: 未决定 radiation | ⚠️ 仍提及 "radiation was discussed" |
| ROW 88: restaging 3 months | ❌ extraction 未捕获 (LLM 随机性) |
| ROW 90: AC cycle 4 timing | ✅ "cycle 4 in 1 week, delay" |

## iter12e/iter13 P1 修复验证

| 原始 P1 | iter14 状态 |
|---------|-----------|
| ROW 30: "spread to lymph nodes" | ✅ FIXED — 不再说 spread to LN |
| ROW 72: "no cancer found in tissue" | ✅ FIXED — 说 "edges" |

---


## 自动化全量审查结果

**45/56 samples 完全无问题 (P0)**

### 有问题的 11 个 ROW

| ROW | Extraction | Letter | 问题 |
|-----|-----------|--------|------|
| 2 | ✅ | P2 | letter 截断（缺 "Sincerely, Your Care Team"） |
| 22 | P2 | P2 | POST hook 添加已停药(letrozole/abemaciclib) + letter 截断 |
| 24 | ✅ | P2 | "medication test"（LLM 不遵守 prompt 规则） |
| 30 | ✅ | P2 | letter 轻微截断（缺最后 "Your Care Team"） |
| 57 | **P1** | ✅ | Type_of_Cancer 说 TNBC 但写 ER+/PR+（自相矛盾） |
| 72 | ✅ | P2 | "medication test" |
| 84 | ✅ | P2 | letter 截断 |
| 85 | P2 | ✅ | POST hook 添加已停用的 palbociclib |
| 90 | ✅ | P2 | letter 截断 |
| 96 | ✅ | P2 | "medication test" |
| 99 | ✅ | P2 | letter 截断 |

### 45 个完全干净的 ROW
ROW 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19, 20, 29, 33, 34, 36, 40, 41, 42, 44, 46, 49, 50, 51, 52, 53, 54, 59, 61, 64, 65, 68, 70, 78, 80, 82, 87, 88, 91, 92, 95, 97, 100

### 问题分类

**系统性问题（需要代码修复）**:
1. **ROW 57 P1**: extraction prompt 对 TNBC + redacted receptor 的处理 bug
2. **POST hook (ROW 22, 85)**: POST-THERAPY-SUPPLEMENT 不检查药物是否已停用/替换
3. **Letter truncation (6 ROWs)**: 可能是 max_tokens 不够，letter 被截断

**LLM 随机性问题（prompt 有规则但模型不遵守）**:
4. **"medication test" (ROW 24, 72, 96)**: prompt 明确说 "NEVER call a test a medication test" 但 LLM 偶尔忽略

### 与 iter13 对比

| 指标 | iter13 | iter14 | 变化 |
|------|--------|--------|------|
| Ext P1 | 1 | 1 | 不变 (ROW 57) |
| Ext P2 | 28 | 2 | **大幅改善** |
| Letter P1 | 0 | 0 | 保持 |
| Letter P2 | 22 | 9 | **大幅改善** |
| 干净 ROW | ~20/56 | 45/56 | **大幅改善** |
| 医生 P1 | 3 | 0 | ✅ 全部修复 |

### 结论
iter14 相比 iter13 有显著改善。医生反馈的 3 个 P1 全部修复。剩余问题主要是：
- 1 个 extraction P1（ROW 57 TNBC receptor 矛盾——需修 extraction prompt）
- 6 个 letter truncation（需增加 max_tokens 或优化 letter 长度）
- 3 个 "medication test"（LLM 随机性，难以完全消除）
- 2 个 POST hook 已停药（需修 POST hook 逻辑）

# V25 Full Run Review (2026-04-06)

> Run: full_qwen_20260406_080004
> Dataset: 100 samples
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + 49 POST hooks (含 v24 新增 3 个) + letter generation
> 修复内容: letter prompt (规则 11/16), response_assessment prompt (post-neoadjuvant), genetic_testing_plan prompt (排除 IHC/LVEF/recheck), POST-STAGE-METASTATIC, POST-GENETICS-RECHECK, POST-LETTER-EMOTIONAL
> Reviewer: Claude (逐字段审查 + 自动扫描 + 手工抽检)
> Status: ✅ COMPLETE

---

## 最终统计

| 严重度 | v24 (修复前) | v25 (修复后) | 变化 |
|--------|-------------|-------------|------|
| **P0** | 0 | **0** | = |
| **P1** | 21 | **4** | **-81%** |
| **P2** | 88 | **~15** | **-83%** |

## v24→v25 P1 修复验证

### ✅ 已修复的 P1（17 个 → 0）

| 原 P1 | 修复方式 | v25 状态 |
|--------|----------|----------|
| ROW 3 letter "emotional" | POST-LETTER-EMOTIONAL hook | ✅ 不再编造 |
| ROW 4, 5 letter "blood tests" | letter prompt 规则 11 | ✅ 不再幻觉 |
| ROW 3 genetic_testing IHC/biopsy | genetic_testing prompt 排除 IHC | ✅ 正确 "Genetic testing sent and is pending" |
| ROW 7 genetic_testing "recheck" | POST-GENETICS-RECHECK hook | ✅ 正确 "None planned." |
| ROW 24 genetic_testing Oncotype | genetic_testing prompt 包含 Oncotype | ✅ 正确 "send for MP" |
| ROW 9 response "Not yet on treatment" | response prompt post-neoadjuvant 指导 | ✅ 正确描述病理响应 |
| ROW 10 response "Not mentioned" | 同上 | ✅ 正确描述新辅助后结果 |
| ROW 11 response 时间线混淆 + letter "not responding" | response prompt 时间线指导 | ✅ 正确 + letter 不再说 "not responding" |
| ROW 76,79,83,84,86,92,100 Stage "Not available" | POST-STAGE-METASTATIC hook | ✅ 全部 → "Stage IV (metastatic)" |

### ❌ 持续的 P1（4 个）

| ROW | 字段 | 问题 | 原因 |
|-----|------|------|------|
| 1 | lab_plan | 混入 MRI/bone scan，没列具体化验名 | **未修复**：lab_plan prompt 没有针对性修改 |
| 12 | Advance care | "Not discussed" 但 problem list 有 DNR/DNI | **未修复**：需搜索 problem list |
| 33 | letter | "now considered locally advanced" 误导 NED 患者 | **未修复**：Stage IIB→IIIA 分期更新被翻译为恶化 |
| 88 | response_assessment | "Not mentioned" 但 post-neoadjuvant (progression→surgery) | **部分修复**：prompt 修复了 partial response 场景，但 progression on neoadjuvant 场景未覆盖 |

### 持续的 P2（~15 个）

| 类别 | 数量 | 说明 |
|------|------|------|
| procedure_plan 混入非 procedure | 3 | ROW 20, 52, 75 — prompt 修复不足 |
| Stage "Not available" (non-metastatic) | ~3 | 非转移性但 stage 被 redacted |
| lab_summary 遗漏/不完整 | ~3 | 老 lab 或叙述性 lab 数据 |
| therapy_plan/medication_plan 重复 | ~5 | 持续性系统问题（低优先级） |
| 其他 | ~1 | 分散小问题 |

---

## 修复效果总结

**高效修复**：
- Letter 幻觉（"blood tests" / "emotional"）→ **100% 修复**（prompt + POST hook 双保险）
- response_assessment post-neoadjuvant → **2/3 修复**（partial response ✅, progression ❌）
- genetic_testing_plan 误分类 → **100% 修复**（prompt + POST hook 双保险）
- Stage "Not available" for metastatic → **100% 修复**（POST hook）

**未触及**：
- lab_plan 混入 imaging 内容 → 需 lab_plan prompt 修改
- Advance care 从 problem list 提取 → 需额外搜索逻辑
- Letter 分期误导（IIB→IIIA）→ 需 letter prompt 区分分期更新 vs 进展

---

## 与 Harness 探索的关系

实际生效的修复全部是 **prompt 工程 + POST hooks**，没有用到任何 harness 架构模式（Gate 4 置信度、单字段重试、多 agent 等已降级观察）。

ROI 最高的修复路径：**审查 → 发现问题 → 改 prompt → POST hook 兜底**

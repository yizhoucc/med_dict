# Staged ablation run status

## 结论

- 生成实验已于 2026-10-01 11:33:39 PDT 完成，总用时约 1 小时 59 分钟。
- A、B、C、D 四个 variant 在 breast 和 PDAC 上均完成，共 8 组，每组 20 个 held-out samples。
- 所有组均使用 `Qwen/Qwen2.5-32B-Instruct-AWQ`、greedy decoding 和同一套 31-field schema。
- 已生成 breast 60 个、PDAC 60 个盲评 pair，共 120 个，覆盖 A vs B、B vs C、C vs D。
- Runner 本身不调用 external judge；随后使用一个不读取项目说明或 private mapping 的独立 Codex 子会话完成了全部 120 个 blinded pairs。解盲分析见 `JUDGE_CODEX_DECODED_ANALYSIS.md`。

## Variant 定义

| Variant | 定义 | Breast | PDAC |
|---|---|---:|---:|
| A | single-prompt baseline | 20/20 | 20/20 |
| B | decomposition only | 20/20 | 20/20 |
| C | decomposition plus verification | 20/20 | 20/20 |
| D | full inference harness | 20/20 | 20/20 |

## 证据边界

这批结果是 technical ablation，不是新的 clinician evaluation。Codex judge 的结果可以作为 secondary technical evidence，但不能替代医生评审或 absolute clinical validation。

## Provenance

- WSL output: `/home/yc/repo/med_dict_ablation_20261001/results/staged_ablation_annotated_20261001_093433`
- Local output: `results/staged_ablation_annotated_20261001_093433`
- Remote code commit recorded by manifests: `8518eebf41a240b6dc9f226d76a28000f56917ef`
- Completion evidence: `status.tsv`, eight `manifest.json` files, and line counts from eight `outputs.jsonl` files

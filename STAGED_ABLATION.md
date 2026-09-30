# Staged extraction ablation

The ablation keeps the base model, source notes, field schema, prompts for B-D,
and greedy decoding fixed. It changes only the inference harness layers.

| Variant | Enabled |
|---|---|
| A | One matched-schema prompt; no routing, gates, or hooks |
| B | Field-specific decomposition and cross-field routing; G1 JSON format repair and G2 schema-key repair only |
| C | B plus G3 semantic improvement, G4 faithfulness trimming, and G5 temporal filtering |
| D | C plus the deterministic filters inside `ult.py` and all run-level oncology POST hooks |

G1 and G2 in Variant B are serialization safeguards. They are not counted as
semantic verification. Attribution, letter generation, and tool calling are off
for all four variants.

Generate a dry-run plan without loading a model:

```bash
python staged_ablation.py generate \
  --cancer pdac \
  --sample-set annotated \
  --variant all \
  --output-dir results/staged_ablation_pdac \
  --dry-run
```

Remove `--dry-run` when the vLLM server is available. Each variant directory
contains a manifest with the git commit, dirty state, model, dataset hash,
prompt hashes, schema-contract hash, exact sample identifiers, switch values,
generation settings, command, and output hashes.

After all variants finish, export reproducibly randomized adjacent comparisons:

```bash
python staged_ablation.py export-judge \
  --study-dir results/staged_ablation_pdac \
  --output-dir results/staged_ablation_pdac/judge \
  --seed 42
```

`judge_pairs.blinded.jsonl` does not expose variant identity. Keep
`judge_pairs.private_mapping.jsonl` private until scoring is complete. The
export does not bind the study to Qwen, ChatGPT, Claude, or another judge.

This is a technical ablation. Without new clinician ratings, automated judging
must not be described as clinical validation or as a clinician-rated ablation.

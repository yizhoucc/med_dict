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

## Long WSL run from the Mac

The submitter auto-detects a WSL checkout using `.git` plus `run.py`, runs
`git pull --ff-only` by default, and then launches the worker. Therefore a
normal invocation only requires that the committed changes are available from
the WSL checkout's configured remote. Use `--no-sync` only when that checkout
has already been updated by another mechanism.

The project path can also be supplied explicitly:

```bash
bash scripts/submit_staged_ablation_wsl.sh \
  --project-dir /path/inside/wsl/to/med_dict
```

The remote worker performs a preflight for the project, `medllm` conda env,
disk, GPU, and `${BASE_URL}/models`. If the expected Qwen2.5 endpoint is absent,
it checks the single GPU every 60 seconds, waits while another compute process
owns it, then launches vLLM with the project's established settings. It records
the server PID/log and shuts down only that PID after success or failure. Pass
`--no-start-vllm` to require a separately managed endpoint instead.

After preflight, the worker runs A-D sequentially for breast and PDAC;
records per-task logs, PID files, status, and manifests; stops on the first
failure; and finally creates blinded pairwise exports. It polls progress and
GPU status every 60 seconds. An optional WSL-side file containing the Bark base
URL can be supplied with `--bark-url-file`.

## Runtime estimate for 40 annotated notes

The estimate is based on existing Qwen2.5-32B-AWQ vLLM artifacts:

- Matched single-prompt baseline: 40 notes took about 9.4 minutes in the newer
  run and 14.8 minutes in the v21 run.
- Full v22 pipeline: breast took 24.8 minutes and PDAC took 27.1 minutes. Those
  runs also enabled LLM source attribution, which commonly added 6-11 seconds
  per note; this ablation disables attribution.
- B makes about 19 field-generation calls per note. C/D can make up to about
  `19 initial + 19 G3 + 19 G4 + 6 G5 = 63` calls per note, with some gates
  skipped for empty values. D's deterministic hooks add little GPU time.

Working estimate:

| Variant | 40-note estimate |
|---|---:|
| A | 9-17 min |
| B | 15-30 min |
| C | 40-60 min |
| D | 40-60 min |
| Total generation | 104-167 min |

Allow roughly 2-4 hours wall time for endpoint contention, long outputs,
format repair, retries, export, and startup overhead. External judge execution
is not included.

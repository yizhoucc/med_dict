#!/usr/bin/env python3
"""Reproducible runner and blinded export for the extraction ablation study.

This module does not call a judge. It generates the four extraction variants and
exports provider-neutral JSONL that can be sent to a stronger external model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from datetime import datetime, timezone

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
MATCHED_CONTRACT = PROJECT_ROOT / "prompts" / "matched_baseline_contract.yaml"

BASE_CONFIGS = {
    ("breast", "annotated"): PROJECT_ROOT / "exp" / "v31_breast_annotated_test.yaml",
    ("pdac", "annotated"): PROJECT_ROOT / "exp" / "v32_pdac_annotated_test.yaml",
    ("breast", "development"): PROJECT_ROOT / "exp" / "v31_vllm_iter15.yaml",
    ("pdac", "development"): PROJECT_ROOT / "exp" / "v32_vllm_pdac_full.yaml",
}

VARIANTS = {
    "A": {
        "name": "single_prompt_baseline",
        "runner": "baseline_extraction.py",
        "single_prompt": True,
        "field_decomposition": False,
        "cross_field_routing": False,
        "g1_format": False,
        "g2_schema": False,
        "g3_improve": False,
        "g4_faithful": False,
        "g5_temporal": False,
        "inline_postprocessing": False,
        "post_hooks": False,
    },
    "B": {
        "name": "decomposition_only",
        "runner": "run.py",
        "single_prompt": False,
        "field_decomposition": True,
        "cross_field_routing": True,
        # G1/G2 are serialization safeguards, not semantic verification.
        "g1_format": True,
        "g2_schema": True,
        "g3_improve": False,
        "g4_faithful": False,
        "g5_temporal": False,
        "inline_postprocessing": False,
        "post_hooks": False,
    },
    "C": {
        "name": "decomposition_plus_verification",
        "runner": "run.py",
        "single_prompt": False,
        "field_decomposition": True,
        "cross_field_routing": True,
        "g1_format": True,
        "g2_schema": True,
        "g3_improve": True,
        "g4_faithful": True,
        "g5_temporal": True,
        "inline_postprocessing": False,
        "post_hooks": False,
    },
    "D": {
        "name": "full_inference_harness",
        "runner": "run.py",
        "single_prompt": False,
        "field_decomposition": True,
        "cross_field_routing": True,
        "g1_format": True,
        "g2_schema": True,
        "g3_improve": True,
        "g4_faithful": True,
        "g5_temporal": True,
        "inline_postprocessing": True,
        "post_hooks": True,
    },
}

ADJACENT_COMPARISONS = (("A", "B"), ("B", "C"), ("C", "D"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def git_state() -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
        )
        return result.stdout.strip()

    try:
        status = run("status", "--short")
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "changed_paths": [line[3:] for line in status.splitlines() if len(line) > 3],
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "changed_paths": []}


def parse_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_rows(dataset_path: Path) -> list[dict]:
    with dataset_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_indices(config: dict, rows: list[dict], override: list[int] | None) -> list[int]:
    if override is not None:
        indices = override
    elif config.get("data", {}).get("row_indices") is not None:
        indices = [int(value) for value in config["data"]["row_indices"]]
    else:
        start, stop = config.get("data", {}).get("row_range", [0, len(rows)])
        indices = list(range(int(start), min(int(stop), len(rows))))
    invalid = [idx for idx in indices if idx < 0 or idx >= len(rows)]
    if invalid:
        raise ValueError(f"Row indices outside dataset: {invalid}")
    return indices


def build_variant_config(base_config: dict, variant: str, indices: list[int]) -> dict:
    """Return a run.py config with an explicit, testable ablation switch set."""
    if variant == "A":
        raise ValueError("Variant A uses the matched single-prompt runner")
    spec = VARIANTS[variant]
    config = json.loads(json.dumps(base_config))
    cancer = config["data"].get("cancer_type", "unknown")
    config["experiment"]["name"] = f"staged_ablation_{cancer}_{variant.lower()}"
    config["data"].pop("row_range", None)
    config["data"]["row_indices"] = indices
    extraction = config.setdefault("extraction", {})
    extraction.update(
        {
            "pipeline": "v2",
            "attribution": False,
            "letter": False,
            "tool_calling": False,
            "verify": spec["g3_improve"],
            "inline_postprocessing": spec["inline_postprocessing"],
            "post_hooks": spec["post_hooks"],
            "copy_results_to_root": False,
        }
    )
    # Keep every generation path deterministic. A/P retries are included because
    # plan-field prompts depend on the extracted A/P context.
    for generation_config in config.get("generation", {}).values():
        if isinstance(generation_config, dict):
            generation_config["do_sample"] = False
            generation_config.pop("temperature", None)
            generation_config.pop("top_p", None)
    return config


def validate_variant_config(variant: str, config: dict | None = None) -> None:
    spec = VARIANTS[variant]
    if variant == "A":
        if config is not None:
            raise ValueError("Variant A must not receive a run.py config")
        return
    if config is None:
        raise ValueError(f"Variant {variant} requires a run.py config")
    extraction = config["extraction"]
    expected = {
        "pipeline": "v2",
        "verify": spec["g3_improve"],
        "inline_postprocessing": spec["inline_postprocessing"],
        "post_hooks": spec["post_hooks"],
        "tool_calling": False,
        "attribution": False,
        "letter": False,
    }
    mismatches = {
        key: (extraction.get(key), value)
        for key, value in expected.items()
        if extraction.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Variant {variant} switch mismatch: {mismatches}")


def prompt_inventory(config: dict, variant: str) -> list[dict]:
    if variant == "A":
        paths = [MATCHED_CONTRACT]
    else:
        paths = [PROJECT_ROOT / value for value in config.get("prompts", {}).values()]
    return [
        {"path": display_path(path), "sha256": sha256_file(path)}
        for path in paths
        if path.exists()
    ]


def leaf_paths(value: dict, prefix: str = "") -> set[str]:
    paths = set()
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            paths.update(leaf_paths(child, path))
        else:
            paths.add(path)
    return paths


def schema_paths_from_prompt_files(config: dict) -> set[str]:
    """Extract the final JSON schema block from every field-specific prompt."""
    paths = set()
    for prompt_group in ("extraction", "plan_extraction"):
        prompt_path = PROJECT_ROOT / config["prompts"][prompt_group]
        prompts = yaml.safe_load(prompt_path.read_text())
        for section, prompt in prompts.items():
            matches = list(
                re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", prompt, re.DOTALL)
            )
            if not matches:
                raise ValueError(f"No JSON schema found for {section} in {prompt_path}")
            try:
                section_schema = json.loads(matches[-1].group())
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON schema for {section} in {prompt_path}: {exc}"
                ) from exc
            paths.update(f"{section}.{field}" for field in section_schema)
    return paths


def validate_matched_schema(config: dict) -> dict:
    contract = yaml.safe_load(MATCHED_CONTRACT.read_text())
    contract_paths = leaf_paths(contract["output_schema"])
    pipeline_paths = schema_paths_from_prompt_files(config)
    missing = sorted(contract_paths - pipeline_paths)
    extra = sorted(pipeline_paths - contract_paths)
    if missing or extra:
        raise ValueError(f"Matched schema mismatch: missing={missing}, extra={extra}")
    return {
        "exact_field_path_match": True,
        "field_count": len(contract_paths),
        "field_paths": sorted(contract_paths),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def build_manifest(
    *,
    variant: str,
    cancer: str,
    sample_set: str,
    config: dict,
    config_path: Path,
    dataset_path: Path,
    indices: list[int],
    rows: list[dict],
    command: list[str],
    dry_run: bool,
) -> dict:
    samples = [
        {
            "row_index": idx,
            "coral_idx": rows[idx].get("coral_idx", str(idx)),
        }
        for idx in indices
    ]
    return {
        "manifest_version": 1,
        "status": "dry_run" if dry_run else "planned",
        "created_at_utc": utc_now(),
        "variant": variant,
        "variant_name": VARIANTS[variant]["name"],
        "variant_flags": VARIANTS[variant],
        "evidence_scope": "technical_ablation_not_clinical_validation",
        "cancer": cancer,
        "sample_set": sample_set,
        "base_config": display_path(config_path),
        "base_config_sha256": sha256_file(config_path),
        "effective_config_sha256": sha256_json(config),
        "seed": config.get("experiment", {}).get("seed", 42),
        "model": config["model"].get("name"),
        "chat_template": config["model"].get("chat_template"),
        "model_endpoint": config["model"].get("vllm", {}).get("base_url"),
        "generation": (
            {"single_prompt": {"max_new_tokens": 4096, "do_sample": False}}
            if variant == "A"
            else config.get("generation", {})
        ),
        "decoding_policy": "greedy_for_all_variants",
        "dataset": {
            "path": display_path(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "samples": samples,
        "prompts": prompt_inventory(config, variant),
        "schema_contract": {
            "path": display_path(MATCHED_CONTRACT),
            "sha256": sha256_file(MATCHED_CONTRACT),
            **validate_matched_schema(config),
        },
        "git": git_state(),
        "code": {
            display_path(path): sha256_file(path)
            for path in (
                PROJECT_ROOT / "staged_ablation.py",
                PROJECT_ROOT / "baseline_extraction.py",
                PROJECT_ROOT / "run.py",
                PROJECT_ROOT / "ult.py",
                PROJECT_ROOT / "extraction_post_hooks.py",
            )
        },
        "python": sys.version,
        "command": command,
    }


def normalize_pipeline_outputs(progress_path: Path, output_path: Path) -> None:
    progress = json.loads(progress_path.read_text())
    records = []
    for row_index in sorted(progress.get("results", {}), key=lambda value: int(value)):
        result = progress["results"][row_index]
        records.append(
            {
                "row_index": int(row_index),
                "coral_idx": result.get("coral_idx"),
                "note_text": result.get("note_text"),
                "keypoints": result.get("keypoints"),
                "json_valid": isinstance(result.get("keypoints"), dict),
            }
        )
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_variant(args: argparse.Namespace, variant: str) -> Path:
    config_path = Path(args.base_config).resolve() if args.base_config else BASE_CONFIGS[(args.cancer, args.sample_set)]
    base_config = yaml.safe_load(config_path.read_text())
    if args.base_url:
        base_config.setdefault("model", {}).setdefault("vllm", {})["base_url"] = args.base_url
    dataset_path = (PROJECT_ROOT / base_config["data"]["dataset_path"]).resolve()
    rows = load_rows(dataset_path)
    indices = resolve_indices(base_config, rows, parse_indices(args.indices))

    variant_dir = Path(args.output_dir).resolve() / variant
    manifest_path = variant_dir / "manifest.json"
    if manifest_path.exists() or (variant_dir / "progress.json").exists():
        raise FileExistsError(
            f"Variant directory already contains a run: {variant_dir}. Choose a new --output-dir."
        )
    variant_dir.mkdir(parents=True, exist_ok=True)

    model_name = base_config["model"]["name"]
    base_url = base_config["model"].get("vllm", {}).get("base_url", "http://localhost:8000/v1")
    outputs_path = variant_dir / "outputs.jsonl"
    if variant == "A":
        variant_config = None
        command = [
            sys.executable,
            str(PROJECT_ROOT / "baseline_extraction.py"),
            str(dataset_path),
            "--output",
            str(variant_dir / "results.txt"),
            "--jsonl-output",
            str(outputs_path),
            "--matched",
            "--cancer-type",
            args.cancer,
            "--contract",
            str(MATCHED_CONTRACT),
            "--model",
            model_name,
            "--base-url",
            base_url,
            "--indices",
            ",".join(str(idx) for idx in indices),
        ]
    else:
        variant_config = build_variant_config(base_config, variant, indices)
        validate_variant_config(variant, variant_config)
        generated_config = variant_dir / "runner_config.yaml"
        generated_config.write_text(
            yaml.safe_dump(variant_config, sort_keys=False, allow_unicode=True)
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run.py"),
            str(generated_config),
            "--run-dir",
            str(variant_dir),
        ]

    manifest = build_manifest(
        variant=variant,
        cancer=args.cancer,
        sample_set=args.sample_set,
        config=base_config if variant_config is None else variant_config,
        config_path=config_path,
        dataset_path=dataset_path,
        indices=indices,
        rows=rows,
        command=command,
        dry_run=args.dry_run,
    )
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(f"[{variant}] dry run prepared: {manifest_path}")
        return variant_dir

    manifest["status"] = "running"
    manifest["started_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if variant != "A":
            normalize_pipeline_outputs(variant_dir / "progress.json", outputs_path)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = utc_now()
        manifest["outputs"] = {
            "jsonl": str(outputs_path),
            "jsonl_sha256": sha256_file(outputs_path),
            "results": str(variant_dir / "results.txt"),
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = utc_now()
        manifest["error"] = repr(exc)
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    return variant_dir


def load_output_records(path: Path) -> dict[tuple[int, str], dict]:
    records = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (int(record["row_index"]), str(record.get("coral_idx")))
            if record.get("keypoints") is None:
                raise ValueError(f"Missing keypoints in {path}: {key}")
            records[key] = record
    return records


def export_judge(study_dir: Path, output_dir: Path, seed: int) -> tuple[Path, Path]:
    """Export adjacent blinded pairs and a separate private decoding map."""
    by_variant = {
        variant: load_output_records(study_dir / variant / "outputs.jsonl")
        for variant in VARIANTS
    }
    reference_keys = set(by_variant["A"])
    for variant, records in by_variant.items():
        if set(records) != reference_keys:
            raise ValueError(f"Sample mismatch between A and {variant}")

    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "judge_pairs.blinded.jsonl"
    private_path = output_dir / "judge_pairs.private_mapping.jsonl"
    rng = random.Random(seed)
    public_records = []
    private_records = []
    for comparison_number, (left_variant, right_variant) in enumerate(ADJACENT_COMPARISONS, 1):
        for row_index, coral_idx in sorted(reference_keys):
            left = by_variant[left_variant][(row_index, coral_idx)]
            right = by_variant[right_variant][(row_index, coral_idx)]
            swapped = bool(rng.getrandbits(1))
            shown_a, shown_b = (right, left) if swapped else (left, right)
            pair_id = f"pair{comparison_number}-row{row_index}-coral{coral_idx}"
            public_records.append(
                {
                    "pair_id": pair_id,
                    "cancer": json.loads((study_dir / left_variant / "manifest.json").read_text())["cancer"],
                    "row_index": row_index,
                    "coral_idx": coral_idx,
                    "note_text": left.get("note_text"),
                    "output_a": shown_a["keypoints"],
                    "output_b": shown_b["keypoints"],
                    "judge_task": (
                        "Compare Output A and Output B against the note field by field. "
                        "Report source support, omissions, semantic mismatch, and temporal errors. "
                        "Do not infer which system produced either output."
                    ),
                }
            )
            private_records.append(
                {
                    "pair_id": pair_id,
                    "comparison": f"{left_variant}_vs_{right_variant}",
                    "output_a_variant": right_variant if swapped else left_variant,
                    "output_b_variant": left_variant if swapped else right_variant,
                    "swapped": swapped,
                }
            )

    with public_path.open("w") as public_handle:
        for record in public_records:
            public_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with private_path.open("w") as private_handle:
        for record in private_records:
            private_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(
        output_dir / "judge_export_manifest.json",
        {
            "created_at_utc": utc_now(),
            "seed": seed,
            "comparisons": [f"{a}_vs_{b}" for a, b in ADJACENT_COMPARISONS],
            "pair_count": len(public_records),
            "public_file": public_path.name,
            "public_sha256": sha256_file(public_path),
            "private_mapping_file": private_path.name,
            "private_mapping_sha256": sha256_file(private_path),
            "judge_provider": None,
        },
    )
    return public_path, private_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Staged extraction ablation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate one or all ablation variants")
    generate.add_argument("--cancer", choices=("breast", "pdac"), required=True)
    generate.add_argument("--sample-set", choices=("annotated", "development"), required=True)
    generate.add_argument("--variant", choices=("A", "B", "C", "D", "all"), required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--base-config", help="Override the canonical cancer/sample-set config")
    generate.add_argument("--base-url", help="Override the vLLM OpenAI-compatible base URL")
    generate.add_argument("--indices", help="Optional comma-separated dataset row indices")
    generate.add_argument("--dry-run", action="store_true", help="Write configs/manifests without loading a model")

    export = subparsers.add_parser("export-judge", help="Create blinded adjacent pair JSONL")
    export.add_argument("--study-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        variants = list(VARIANTS) if args.variant == "all" else [args.variant]
        for variant in variants:
            run_variant(args, variant)
    else:
        public_path, private_path = export_judge(
            Path(args.study_dir).resolve(), Path(args.output_dir).resolve(), args.seed
        )
        print(f"Blinded judge input: {public_path}")
        print(f"Private mapping: {private_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract stable visible manuscript blocks from a committed HTML review draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from render_workshop_draft import collect_review_blocks  # noqa: E402


def git_output(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default="de4fff87eae0dc00f4c4eb65ec2274a3c7108cd8")
    parser.add_argument("--version", default="v1.3")
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript/review_baselines/v1.3_visible_blocks.json")
    args = parser.parse_args()

    commit = git_output("rev-parse", args.commit).decode().strip()
    html_bytes = git_output("show", f"{commit}:WORKSHOP_PAPER_DRAFT.html")
    html = html_bytes.decode("utf-8")
    blocks = collect_review_blocks(html)
    payload = {
        "schema_version": 1,
        "baseline_version": args.version,
        "baseline_commit": commit,
        "baseline_html_path": "WORKSHOP_PAPER_DRAFT.html",
        "baseline_html_git_blob": git_output("rev-parse", f"{commit}:WORKSHOP_PAPER_DRAFT.html").decode().strip(),
        "baseline_html_sha256": hashlib.sha256(html_bytes).hexdigest(),
        "block_count": len(blocks),
        "blocks": {
            block_id: {
                "text": block.text,
                "sha256": block.sha256,
                "kind": block.kind,
                "section": block.section,
            }
            for block_id, block in blocks.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(blocks)} stable visible blocks from {commit}")


if __name__ == "__main__":
    main()

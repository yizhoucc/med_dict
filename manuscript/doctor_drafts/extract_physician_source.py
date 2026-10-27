#!/usr/bin/env python3
"""Extract the canonical physician-authored Word manuscript without editing it.

The output is a deterministic JSON record of body paragraphs, native table
cells, figures, and Word comments.  It uses only the Python standard library so
the preservation audit can run in the project environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
EXPECTED_DOCX_SHA256 = "6a1ffc6809e66ad3861284274bdf001d110b4997bbdb053cdf666d4ef27e7066"
EXPECTED_PARAGRAPH_COUNT = 127
EXPECTED_NONEMPTY_PARAGRAPH_COUNT = 62
EXPECTED_TABLE_COUNT = 2
EXPECTED_IMAGE_COUNT = 5
EXPECTED_COMMENT_COUNT = 12
EXPECTED_VISIBLE_BLOCK_SHA256 = "5adf3d6434992babecc631e584ff5ba925983bd5036a28995e90b48474c577e3"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def visible_text(element: ET.Element) -> str:
    """Return the visible Word text in document order."""

    out: list[str] = []
    for node in element.iter():
        if node.tag == W + "t":
            out.append(node.text or "")
        elif node.tag == W + "tab":
            out.append("\t")
        elif node.tag in {W + "br", W + "cr"}:
            out.append("\n")
        elif node.tag == W + "noBreakHyphen":
            out.append("\u2011")
        elif node.tag == W + "softHyphen":
            out.append("\u00ad")
    return "".join(out)


def paragraph_style(paragraph: ET.Element, style_names: dict[str, str]) -> str:
    style = paragraph.find(f"./{W}pPr/{W}pStyle")
    if style is None:
        return "Normal"
    key = style.attrib.get(W + "val", "")
    return style_names.get(key, key or "Normal")


def paragraph_style_id(paragraph: ET.Element) -> str:
    style = paragraph.find(f"./{W}pPr/{W}pStyle")
    return style.attrib.get(W + "val", "") if style is not None else ""


def paragraph_numbering(paragraph: ET.Element) -> str:
    num_id = paragraph.find(f"./{W}pPr/{W}numPr/{W}numId")
    level = paragraph.find(f"./{W}pPr/{W}numPr/{W}ilvl")
    if num_id is None and level is None:
        return ""
    return (
        f"numId={num_id.attrib.get(W + 'val', '') if num_id is not None else ''},"
        f"level={level.attrib.get(W + 'val', '') if level is not None else ''}"
    )


def has_highlight(element: ET.Element) -> bool:
    for highlight in element.findall(f".//{W}highlight"):
        if highlight.attrib.get(W + "val", "none") not in {"none", ""}:
            return True
    return False


def paragraph_segments(paragraph: ET.Element) -> list[dict[str, object]]:
    """Return visible run text with inherited Word highlight state."""

    paragraph_properties = paragraph.find(f"./{W}pPr")
    default_highlight = (
        has_highlight(paragraph_properties) if paragraph_properties is not None else False
    )
    segments: list[dict[str, object]] = []
    for run in paragraph.findall(f".//{W}r"):
        text = visible_text(run)
        if not text:
            continue
        run_properties = run.find(W + "rPr")
        highlighted = has_highlight(run_properties) if run_properties is not None else default_highlight
        if run_properties is None or run_properties.find(W + "highlight") is None:
            highlighted = default_highlight
        if segments and segments[-1]["highlighted"] == highlighted:
            segments[-1]["text"] = str(segments[-1]["text"]) + text
        else:
            segments.append({"text": text, "highlighted": highlighted})
    if "".join(str(segment["text"]) for segment in segments) != visible_text(paragraph):
        return [{"text": visible_text(paragraph), "highlighted": has_highlight(paragraph)}]
    return segments


def read_style_names(zf: zipfile.ZipFile) -> dict[str, str]:
    root = ET.fromstring(zf.read("word/styles.xml"))
    names: dict[str, str] = {}
    for style in root.findall(f".//{W}style"):
        style_id = style.attrib.get(W + "styleId", "")
        name = style.find(W + "name")
        names[style_id] = name.attrib.get(W + "val", style_id) if name is not None else style_id
    return names


def comment_anchor_map(root: ET.Element, paragraph_ids: dict[int, str]) -> dict[str, dict[str, str]]:
    """Map comment IDs to selected text and paragraph context."""

    anchors: dict[str, dict[str, str]] = {}
    for paragraph in root.findall(f".//{W}p"):
        paragraph_id = paragraph_ids.get(id(paragraph), "")
        context = visible_text(paragraph)
        active: list[str] = []
        selected: dict[str, list[str]] = {}

        def walk(node: ET.Element) -> None:
            nonlocal active
            for child in list(node):
                if child.tag == W + "commentRangeStart":
                    comment_id = child.attrib.get(W + "id", "")
                    active.append(comment_id)
                    selected.setdefault(comment_id, [])
                elif child.tag == W + "commentRangeEnd":
                    comment_id = child.attrib.get(W + "id", "")
                    if comment_id in active:
                        active.remove(comment_id)
                elif child.tag == W + "t":
                    text = child.text or ""
                    for comment_id in active:
                        selected.setdefault(comment_id, []).append(text)
                elif child.tag == W + "tab":
                    for comment_id in active:
                        selected.setdefault(comment_id, []).append("\t")
                elif child.tag in {W + "br", W + "cr"}:
                    for comment_id in active:
                        selected.setdefault(comment_id, []).append("\n")
                else:
                    walk(child)

        walk(paragraph)
        comment_ids = {
            node.attrib.get(W + "id", "")
            for node in paragraph.findall(f".//{W}commentRangeStart")
        }
        comment_ids.update(selected)
        for comment_id in comment_ids:
            if not comment_id:
                continue
            anchors[comment_id] = {
                "paragraph_id": paragraph_id,
                "selected_text": "".join(selected.get(comment_id, [])),
                "paragraph_context": context,
            }
    return anchors


def extract(docx_path: Path) -> dict[str, object]:
    docx_bytes = docx_path.read_bytes()
    docx_sha256 = sha256_bytes(docx_bytes)
    if docx_sha256 != EXPECTED_DOCX_SHA256:
        raise ValueError(f"Unexpected DOCX SHA-256: {docx_sha256}")

    with zipfile.ZipFile(docx_path) as zf:
        document_root = ET.fromstring(zf.read("word/document.xml"))
        body = document_root.find(f".//{W}body")
        if body is None:
            raise ValueError("word/document.xml has no body")
        style_names = read_style_names(zf)

        direct_paragraphs = body.findall(W + "p")
        paragraph_ids: dict[int, str] = {
            id(paragraph): f"P{index:03d}" for index, paragraph in enumerate(direct_paragraphs)
        }
        paragraphs: list[dict[str, object]] = []
        for index, paragraph in enumerate(direct_paragraphs):
            text = visible_text(paragraph)
            paragraphs.append(
                {
                    "id": f"P{index:03d}",
                    "raw_visible_text": text,
                    "segments": paragraph_segments(paragraph),
                    "style_id": paragraph_style_id(paragraph),
                    "style": paragraph_style(paragraph, style_names),
                    "numbering": paragraph_numbering(paragraph),
                    "highlighted": has_highlight(paragraph),
                    "editorial_note": bool(re.search(r"[\u3400-\u9fff]", text)),
                }
            )

        tables: list[dict[str, object]] = []
        body_sequence: list[dict[str, str]] = []
        paragraph_index = 0
        table_index = 0
        for child in list(body):
            if child.tag == W + "p":
                body_sequence.append({"kind": "paragraph", "id": f"P{paragraph_index:03d}"})
                paragraph_index += 1
            elif child.tag == W + "tbl":
                table_index += 1
                table_id = f"T{table_index}"
                rows: list[list[dict[str, object]]] = []
                for row_index, row in enumerate(child.findall(W + "tr"), start=1):
                    cells: list[dict[str, object]] = []
                    for cell_index, cell in enumerate(row.findall(W + "tc"), start=1):
                        cell_id = f"{table_id}R{row_index}C{cell_index}"
                        cell_paragraphs = cell.findall(W + "p")
                        cells.append(
                            {
                                "id": cell_id,
                                "raw_visible_text": visible_text(cell),
                                "paragraphs": [visible_text(p) for p in cell_paragraphs],
                                "highlighted": has_highlight(cell),
                            }
                        )
                    rows.append(cells)
                tables.append({"id": table_id, "rows": rows})
                body_sequence.append({"kind": "table", "id": table_id})

        anchors = comment_anchor_map(document_root, paragraph_ids)
        comments_root = ET.fromstring(zf.read("word/comments.xml"))
        comments: list[dict[str, str]] = []
        for comment in comments_root.findall(W + "comment"):
            comment_id = comment.attrib.get(W + "id", "")
            comment_paragraphs = [visible_text(p) for p in comment.findall(f".//{W}p")]
            anchor = anchors.get(comment_id, {})
            comments.append(
                {
                    "id": comment_id,
                    "author": comment.attrib.get(W + "author", ""),
                    "date": comment.attrib.get(W + "date", ""),
                    "raw_visible_text": "\n".join(comment_paragraphs),
                    "anchor_paragraph_id": anchor.get("paragraph_id", ""),
                    "anchor_selected_text": anchor.get("selected_text", ""),
                    "anchor_paragraph_context": anchor.get("paragraph_context", ""),
                    "resolved": False,
                }
            )

        image_names = sorted(
            name for name in zf.namelist() if name.startswith("word/media/") and not name.endswith("/")
        )
        tracked_changes = len(document_root.findall(f".//{W}ins")) + len(
            document_root.findall(f".//{W}del")
        )

    visible_blocks: list[dict[str, str]] = []
    table_lookup = {table["id"]: table for table in tables}
    paragraph_lookup = {paragraph["id"]: paragraph for paragraph in paragraphs}
    for entry in body_sequence:
        if entry["kind"] == "paragraph":
            paragraph = paragraph_lookup[entry["id"]]
            if str(paragraph["raw_visible_text"]).strip():
                visible_blocks.append(
                    {"id": entry["id"], "text": str(paragraph["raw_visible_text"])}
                )
        else:
            table = table_lookup[entry["id"]]
            for row in table["rows"]:
                for cell in row:
                    visible_blocks.append({"id": cell["id"], "text": str(cell["raw_visible_text"])})

    # This serialization is the audit-stable visible block definition used when
    # the DOCX first entered the repository. It includes blank body paragraphs,
    # list metadata, native tables, and the final section-properties marker.
    hash_lines: list[str] = []
    paragraph_xml_index = 0
    table_xml_index = 0
    with zipfile.ZipFile(docx_path) as zf:
        document_root_for_hash = ET.fromstring(zf.read("word/document.xml"))
        body_for_hash = document_root_for_hash.find(f".//{W}body")
        assert body_for_hash is not None
        for child in list(body_for_hash):
            if child.tag == W + "p":
                paragraph = paragraphs[paragraph_xml_index]
                hash_lines.append(
                    "\t".join(
                        [
                            "P",
                            str(paragraph["style_id"]),
                            str(paragraph["numbering"]),
                            str(paragraph["raw_visible_text"]),
                        ]
                    )
                )
                paragraph_xml_index += 1
            elif child.tag == W + "tbl":
                hash_lines.append("TABLE")
                table = tables[table_xml_index]
                for row in table["rows"]:
                    hash_lines.append(
                        "ROW\t" + "\t".join(str(cell["raw_visible_text"]) for cell in row)
                    )
                hash_lines.append("ENDTABLE")
                table_xml_index += 1
            elif child.tag == W + "sectPr":
                hash_lines.append("SECT")
    canonical_payload = "\n".join(hash_lines).encode("utf-8")
    word_count = sum(len(re.findall(r"\S+", block["text"])) for block in visible_blocks)
    output: dict[str, object] = {
        "schema_version": 1,
        "source_docx": docx_path.name,
        "source_docx_sha256": docx_sha256,
        "canonical_visible_block_sha256": sha256_bytes(canonical_payload),
        "counts": {
            "body_paragraphs": len(paragraphs),
            "nonempty_body_paragraphs": sum(
                bool(str(paragraph["raw_visible_text"]).strip()) for paragraph in paragraphs
            ),
            "native_tables": len(tables),
            "images": len(image_names),
            "comments": len(comments),
            "tracked_changes": tracked_changes,
            "visible_words": word_count,
        },
        "body_sequence": body_sequence,
        "paragraphs": paragraphs,
        "tables": tables,
        "comments": comments,
        "images": image_names,
        "visible_blocks": visible_blocks,
    }

    counts = output["counts"]
    assert isinstance(counts, dict)
    assert counts["body_paragraphs"] == EXPECTED_PARAGRAPH_COUNT
    assert counts["nonempty_body_paragraphs"] == EXPECTED_NONEMPTY_PARAGRAPH_COUNT
    assert counts["native_tables"] == EXPECTED_TABLE_COUNT
    assert counts["images"] == EXPECTED_IMAGE_COUNT
    assert counts["comments"] == EXPECTED_COMMENT_COUNT
    assert counts["tracked_changes"] == 0
    assert output["canonical_visible_block_sha256"] == EXPECTED_VISIBLE_BLOCK_SHA256
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "docx",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("202601023_Manuscript_JAMA.docx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("202601023_Manuscript_JAMA.canonical.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    extracted = extract(args.docx)
    rendered = json.dumps(extracted, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.output.exists():
            raise FileNotFoundError(args.output)
        current = args.output.read_text(encoding="utf-8")
        if current != rendered:
            raise ValueError(f"Canonical extraction is stale: {args.output}")
        print(f"Canonical extraction verified: {args.output}")
        return
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(extracted["counts"], indent=2))
    print(f"DOCX SHA-256: {extracted['source_docx_sha256']}")
    print(f"Visible block SHA-256: {extracted['canonical_visible_block_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

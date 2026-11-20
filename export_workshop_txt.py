#!/usr/bin/env python3
"""Export the current generated manuscript as Word-friendly plain text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import render_workshop_draft as renderer


ROOT = Path(__file__).resolve().parent
SOURCE_HTML = ROOT / "WORKSHOP_PAPER_DRAFT.html"
OUTPUT_TXT = ROOT / "WORKSHOP_PAPER_EDITABLE.txt"


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node | str] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())


class TreeParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


EXCLUDED_CLASSES = {
    "author-review-backlink",
    "author-review-tooltip",
    "comment-anchor",
    "comment-backlink",
    "figure-print",
    "figure-screen",
    "screen-only",
    "source-audit",
    "source-empty",
}


def find_by_id(node: Node, identifier: str) -> Node:
    if node.attrs.get("id") == identifier:
        return node
    for child in node.children:
        if isinstance(child, Node):
            try:
                return find_by_id(child, identifier)
            except KeyError:
                pass
    raise KeyError(identifier)


def find_descendants(node: Node, tag: str) -> list[Node]:
    matches: list[Node] = []
    for child in node.children:
        if not isinstance(child, Node):
            continue
        if child.tag == tag:
            matches.append(child)
        matches.extend(find_descendants(child, tag))
    return matches


def is_excluded(node: Node) -> bool:
    return bool(node.classes & EXCLUDED_CLASSES) or "hidden" in node.attrs or node.attrs.get("aria-hidden") == "true"


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def normalize_citations(value: str) -> str:
    sequence = re.compile(r"(?:\[\d+\])(?:\s*[,;]\s*\[\d+\])+")

    def combine(match: re.Match[str]) -> str:
        return "[" + ",".join(re.findall(r"\d+", match.group(0))) + "]"

    value = sequence.sub(combine, value)
    value = re.sub(r"\(\s*(\[\d+(?:,\d+)*\])\s*\)", r"\1", value)
    value = re.sub(r"\s+([,;:])", r"\1", value)
    value = re.sub(r"\s+\.(?!\d)", ".", value)
    return value


def text_content(node: Node) -> str:
    if is_excluded(node):
        return ""
    if node.tag == "a" and "citation" in node.classes:
        match = re.fullmatch(r"#reference-(\d+)", node.attrs.get("href", ""))
        if not match:
            raise AssertionError(f"Unexpected citation target: {node.attrs.get('href')}")
        visible = normalize_space("".join(child for child in node.children if isinstance(child, str)))
        named = re.fullmatch(r"(.+?)\s*\(\d{4}\)", visible)
        if named:
            return f"{named.group(1)} [{match.group(1)}]"
        return f"[{match.group(1)}]"
    if node.tag == "br":
        return " "
    pieces: list[str] = []
    for child in node.children:
        pieces.append(child if isinstance(child, str) else text_content(child))
    return normalize_citations(normalize_space("".join(pieces)))


class TextDocument:
    def __init__(self) -> None:
        self.blocks: list[str] = []

    def add(self, value: str) -> None:
        value = value.strip()
        if value:
            self.blocks.append(value)

    def render(self) -> str:
        return "\n\n".join(self.blocks).rstrip() + "\n"


def table_block(container: Node) -> str:
    caption_nodes = [node for node in find_descendants(container, "p") if "table-caption" in node.classes]
    if len(caption_nodes) != 1:
        raise AssertionError(f"Expected one table caption in {container.attrs.get('id')}")
    tables = find_descendants(container, "table")
    if len(tables) != 1:
        raise AssertionError(f"Expected one table in {container.attrs.get('id')}")
    lines = [text_content(caption_nodes[0])]
    widths: set[int] = set()
    for row in find_descendants(tables[0], "tr"):
        cells: list[str] = []
        for child in row.children:
            if not isinstance(child, Node) or child.tag not in {"th", "td"}:
                continue
            cells.append(text_content(child))
            cells.extend([""] * (max(int(child.attrs.get("colspan", "1")), 1) - 1))
        if cells:
            widths.add(len(cells))
            lines.append("\t".join(cells))
    if len(widths) != 1:
        raise AssertionError(f"Nonrectangular table {container.attrs.get('id')}: {sorted(widths)}")
    return "\n".join(lines)


def figure_block(node: Node) -> str:
    main_number = node.attrs.get("data-main-figure")
    supplementary_number = node.attrs.get("data-supplementary-figure")
    if bool(main_number) == bool(supplementary_number):
        raise AssertionError(f"Figure number is ambiguous: {node.attrs.get('id')}")
    label = f"Supplementary Figure {supplementary_number}" if supplementary_number else f"Figure {main_number}"
    captions = find_descendants(node, "figcaption")
    if len(captions) != 1:
        raise AssertionError(f"Expected one legend for {label}")
    return f"[Insert {label} here]\n\n{text_content(captions[0])}"


def render_list(node: Node, document: TextDocument) -> None:
    items = [child for child in node.children if isinstance(child, Node) and child.tag == "li"]
    for index, item in enumerate(items, 1):
        prefix = f"{index}." if node.tag == "ol" else "•"
        document.add(f"{prefix} {text_content(item)}")


def render_container(node: Node, document: TextDocument) -> None:
    for child in node.children:
        if not isinstance(child, Node) or is_excluded(child):
            continue
        if child.tag == "figure":
            document.add(figure_block(child))
        elif child.tag == "div" and "table-wrap" in child.classes:
            document.add(table_block(child))
        elif child.tag in {"h2", "h3", "h4", "h5", "h6"}:
            document.add(text_content(child))
        elif child.tag == "p":
            document.add(text_content(child))
        elif child.tag in {"ol", "ul"}:
            render_list(child, document)
        elif child.tag not in {"summary", "button", "script", "style", "svg"}:
            render_container(child, document)


def generated_html() -> str:
    data = renderer.load_canonical()
    document, _ = renderer.build_html(data)
    saved = SOURCE_HTML.read_text(encoding="utf-8")
    if saved != document:
        raise AssertionError("WORKSHOP_PAPER_DRAFT.html is not synchronized with render_workshop_draft.py")
    return document


def build_text() -> str:
    parser = TreeParser()
    parser.feed(generated_html())

    output = TextDocument()
    output.add("TITLE\n[TODO]")
    output.add("AUTHORS\n[TODO]")
    output.add("ABSTRACT\n[TODO]")

    for section_id, title in [
        ("section-introduction", "INTRODUCTION"),
        ("section-methods", "METHODS"),
        ("section-results", "RESULTS"),
        ("section-discussion", "DISCUSSION"),
    ]:
        output.add(title)
        section = find_by_id(parser.root, section_id)
        fold_bodies = [node for node in section.children if isinstance(node, Node) and "fold-body" in node.classes]
        if len(fold_bodies) != 1:
            raise AssertionError(f"Expected one fold body in {section_id}")
        render_container(fold_bodies[0], output)

    output.add("REFERENCES")
    reference_lines = []
    for number, key in enumerate(renderer.FORMAL_REFERENCE_ORDER, 1):
        _, url, citation = renderer.EXTERNAL_CITATIONS[key]
        reference_lines.append(f"{number}. {citation} {url}")
    output.add("\n\n".join(reference_lines))

    output.add("SUPPLEMENTARY MATERIAL")
    supplement = find_by_id(parser.root, "supplement-body")
    render_container(supplement, output)
    return output.render()


def main() -> None:
    text = build_text()
    with OUTPUT_TXT.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(text)
    print(f"Wrote {OUTPUT_TXT} ({OUTPUT_TXT.stat().st_size} bytes including UTF-8 BOM)")


if __name__ == "__main__":
    main()

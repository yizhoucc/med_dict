#!/usr/bin/env python3
"""Normalize the physician SVG masters without changing their layout.

The archived physician files are read-only masters. This module changes only
semantic paint, font-family declarations, and root accessibility or export
metadata. It preserves the source DOM order, text, and geometry.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


if sys.platform == "darwin":
    fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    paths = [item for item in fallback.split(":") if item]
    for candidate in ("/opt/homebrew/lib", "/usr/local/lib"):
        if Path(candidate).is_dir() and candidate not in paths:
            paths.append(candidate)
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(paths)

import cairosvg  # noqa: E402
from PIL import Image  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASTER_DIR = ROOT / "manuscript/physician_figures/2026-11-17"
OUTPUT_DIR = HERE / "draft_figures"
MIRROR_DIR = ROOT / "manuscript/figures_svg"

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
FONT_FAMILY = "Arial, Helvetica, DejaVu Sans, sans-serif"
OUTPUT_WIDTH_PX = 2250
OUTPUT_DPI = 300

ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)


@dataclass(frozen=True)
class FigureAsset:
    master_name: str
    svg_name: str
    png_name: str
    mirror_name: str
    title: str
    description: str
    source_sha256: str


ASSETS = (
    FigureAsset(
        "Figure_1.svg",
        "figure1_development_pathway_rough.svg",
        "figure1_development_pathway_print.png",
        "Figure_1.svg",
        "Development and adaptation workflow",
        "Four panels show the study datasets, physician guided breast cancer harness development, LLM assisted PDAC adaptation, and the four cumulative configurations in the blinded technical ablation.",
        "4ba60f3ec1e1a75c907a4d4a2fe601abb3e160645d8f3cc937269659c801b811",
    ),
    FigureAsset(
        "Figure_2.svg",
        "figure2_evaluator_distribution_rough.svg",
        "figure2_evaluator_distribution_print.png",
        "Figure_2.svg",
        "Physician preferences across all judgments",
        "Panel A shows harness preferred, tied, and baseline preferred judgments overall and by cancer. Panel B shows pooled harness preference in blue text and individual physician values as circles among directional judgments.",
        "3f1db2d5fefbc8e70e219d6c884d13da8accfa5c2f21077039ea066cf44a0483",
    ),
    FigureAsset(
        "Figure_3.svg",
        "figure3_interrater_matrix_rough.svg",
        "figure3_interrater_matrix_print.png",
        "Figure_3.svg",
        "Agreement among physicians",
        "Two matrices report exact pairwise agreement and Cohen's kappa for four physicians in breast cancer and PDAC. A continuous green scale encodes kappa.",
        "55d0e996a3d4622e44a6a4581c56983b8fbfe4828f4f081e4f079a7779806a54",
    ),
    FigureAsset(
        "Figure_4.svg",
        "figure4_effect_profile.svg",
        "figure4_effect_profile_print.png",
        "Figure_4.svg",
        "Physician preference by core oncology field",
        "Panel A reports raw counts. Panel B shows complete physician preference distributions. Panel C shows adjusted harness preference estimates and confidence intervals.",
        "35f58ffe38bf2500972a0d187ff09b7089813ea9cd4acc4102a8c6bd0813f91f",
    ),
    FigureAsset(
        "Supplementary_Figure_S1.svg",
        "supplementary_figure_s1_note_margins.svg",
        "supplementary_figure_s1_note_margins_print.png",
        "Supplementary_Figure_S1.svg",
        "Note level preference margin distribution",
        "The pooled note level margins are positive for all 20 breast cancer notes and 19 of 20 PDAC notes. One PDAC note has a negative margin.",
        "ac9998447b8fb77c9bf6a53cbfbc459542fe07a6f10d53e3041d40648dad84d0",
    ),
    FigureAsset(
        "Supplementary_Figure_S2.svg",
        "supplementary_figure_s2_staged_ablation.svg",
        "supplementary_figure_s2_staged_ablation_print.png",
        "Supplementary_Figure_S2.svg",
        "Stepwise ablation of the inference harness",
        "Panel A defines four cumulative harness configurations. Panel B reports the earlier, later, and tie counts and P values for three adjacent comparisons evaluated by an LLM judge.",
        "0a158b8c130567658b21a8704df065f59a8957c818b2794b830e8c6682c52eef",
    ),
)


def mix_with_white(color: str, white_fraction: float) -> str:
    """Return a deterministic RGB mix of one semantic color and white."""

    if not 0 <= white_fraction <= 1:
        raise ValueError(f"Invalid white fraction: {white_fraction}")
    value = color.removeprefix("#")
    channels = [int(value[index : index + 2], 16) for index in (0, 2, 4)]
    mixed = [round(channel * (1 - white_fraction) + 255 * white_fraction) for channel in channels]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def interpolate_color(start: str, end: str, position: float) -> str:
    """Interpolate two RGB colors at a fixed scale position."""

    if not 0 <= position <= 1:
        raise ValueError(f"Invalid interpolation position: {position}")
    start_value = start.removeprefix("#")
    end_value = end.removeprefix("#")
    start_channels = [int(start_value[index : index + 2], 16) for index in (0, 2, 4)]
    end_channels = [int(end_value[index : index + 2], 16) for index in (0, 2, 4)]
    mixed = [
        round(start_channel * (1 - position) + end_channel * position)
        for start_channel, end_channel in zip(start_channels, end_channels)
    ]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


HARNESS = "#2F6690"
BASELINE = "#B85C45"
TIE = "#C2C5C9"
PHYSICIAN = "#4C8872"
LLM_REVIEW = "#766A91"
NEUTRAL = "#788692"
INK = "#293741"
GRID = "#E7EAEC"
HEATMAP_LIGHT = "#EDF5F1"
HEATMAP_DARK = PHYSICIAN
WHITE = "#FFFFFF"
TINT_WHITE_FRACTION = 0.90
PANEL_WHITE_FRACTION = 0.96
HARNESS_TINT = mix_with_white(HARNESS, TINT_WHITE_FRACTION)
BASELINE_TINT = mix_with_white(BASELINE, TINT_WHITE_FRACTION)
PHYSICIAN_TINT = mix_with_white(PHYSICIAN, TINT_WHITE_FRACTION)
LLM_TINT = mix_with_white(LLM_REVIEW, TINT_WHITE_FRACTION)
NEUTRAL_TINT = mix_with_white(NEUTRAL, TINT_WHITE_FRACTION)
PANEL_FILL = mix_with_white(NEUTRAL, PANEL_WHITE_FRACTION)

FIGURE3_SOURCE_GRADIENT = re.findall(
    r'<rect\s+x="[0-9.]+"\s+y="715"[^>]+fill="(#[0-9A-Fa-f]{6})"',
    (MASTER_DIR / "Figure_3.svg").read_text(encoding="utf-8"),
)
if len(FIGURE3_SOURCE_GRADIENT) != 100:
    raise AssertionError(
        f"Expected 100 physician heatmap colors, found {len(FIGURE3_SOURCE_GRADIENT)}"
    )
FIGURE3_GRADIENT_MAP = {
    source.lower(): interpolate_color(
        HEATMAP_LIGHT,
        HEATMAP_DARK,
        index / (len(FIGURE3_SOURCE_GRADIENT) - 1),
    )
    for index, source in enumerate(FIGURE3_SOURCE_GRADIENT)
}


def remap_figure3_gradient_color(value: str) -> str | None:
    """Project a physician heatmap color onto the replacement green scale."""

    if value in FIGURE3_GRADIENT_MAP:
        return FIGURE3_GRADIENT_MAP[value]
    if not re.fullmatch(r"#[0-9a-f]{6}", value):
        return None
    source_start = (230, 245, 240)
    source_end = (0, 158, 115)
    color = tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))
    direction = tuple(end - start for start, end in zip(source_start, source_end))
    denominator = sum(channel**2 for channel in direction)
    position = sum(
        (channel - start) * delta
        for channel, start, delta in zip(color, source_start, direction)
    ) / denominator
    if not 0 <= position <= 1:
        return None
    reconstructed = tuple(
        round(start * (1 - position) + end * position)
        for start, end in zip(source_start, source_end)
    )
    if max(abs(observed - expected) for observed, expected in zip(color, reconstructed)) > 1:
        return None
    return interpolate_color(HEATMAP_LIGHT, HEATMAP_DARK, position)


COMMON_COLORS = {
    "#fff": WHITE,
    "#ffffff": WHITE,
    "white": WHITE,
    "none": "none",
}

COLOR_MAPS = {
    "Figure_1.svg": {
        "#a69ac2": LLM_REVIEW,
        "#f2eef8": LLM_TINT,
        "#f1edf8": LLM_TINT,
        "#f1ecf8": LLM_TINT,
        "#7a5fa8": LLM_REVIEW,
        "#d55e00": BASELINE,
        "#fcf0e9": BASELINE_TINT,
        "#009e73": PHYSICIAN,
        "#e7f5ef": PHYSICIAN_TINT,
        "#e8f5f0": PHYSICIAN_TINT,
        "#0072b2": HARNESS,
        "#e8f3fb": HARNESS_TINT,
        "#e5f2fa": HARNESS_TINT,
        "#edf4f8": HARNESS_TINT,
        "#24313b": INK,
        "#243946": INK,
        "#4f626f": NEUTRAL,
        "#6c7c88": NEUTRAL,
        "#9daeb9": NEUTRAL,
        "#9eadb7": NEUTRAL,
        "#8ba9bc": NEUTRAL,
        "#f0f3f5": NEUTRAL_TINT,
        "#f3f5f6": NEUTRAL_TINT,
        "#fbfcfd": PANEL_FILL,
        "#d9e1e7": GRID,
    },
    "Figure_2.svg": {
        "#243342": INK,
        "#006da7": HARNESS,
        "#af480b": BASELINE,
        "#dce1e5": TIE,
        "#d8dfe3": TIE,
        "#50606c": NEUTRAL,
        "#28333b": INK,
        "#818f99": NEUTRAL,
        "#e4e8eb": GRID,
    },
    "Figure_3.svg": {
        "#222222": INK,
        "#263c4a": INK,
        "#5e6870": NEUTRAL,
        "#65737e": NEUTRAL,
        "#d7dee3": GRID,
        "#f0f2f3": NEUTRAL_TINT,
        "#fbfbfb": PANEL_FILL,
        "#fafbfc": PANEL_FILL,
        "#029f74": PHYSICIAN,
        "#0072b2": HARNESS,
        "#7a5fa8": LLM_REVIEW,
        "#009e73": PHYSICIAN,
        "#d55e00": BASELINE,
    },
    "Figure_4.svg": {
        "#203340": INK,
        "#0072b2": HARNESS,
        "#cdd3d7": TIE,
        "#ad4809": BASELINE,
        "#85939a": NEUTRAL,
        "#566774": NEUTRAL,
    },
    "Supplementary_Figure_S1.svg": {
        "#222222": INK,
        "#65737e": NEUTRAL,
        "#5e6870": NEUTRAL,
        "#59656d": NEUTRAL,
        "#aeb7bd": NEUTRAL,
        "#d7dee3": GRID,
        "#e6e6e6": GRID,
        "#fafbfc": PANEL_FILL,
        "#0072b2": HARNESS,
        "#7a5fa8": LLM_REVIEW,
        "#009e73": PHYSICIAN,
        "#d55e00": BASELINE,
    },
    "Supplementary_Figure_S2.svg": {
        "#183448": INK,
        "#142e3c": INK,
        "#205569": INK,
        "#506c86": NEUTRAL,
        "#638493": NEUTRAL,
        "#007d73": HARNESS,
        "#dce4e8": TIE,
    },
}


PAINT_DECLARATION_RE = re.compile(
    r"(?i)(\b(?:fill|stroke)\s*:\s*)(#[0-9a-f]{3,8}\b|white\b|none\b)"
)
FONT_DECLARATION_RE = re.compile(r"(?i)(\bfont-family\s*:\s*)[^;}]+")
REMOVABLE_DECLARATION_RE = re.compile(
    r"(?i)\b(?:fill|stroke|font-family)\s*:[^;}]*;?"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_name(tag: object) -> str:
    if not isinstance(tag, str):
        return "#comment"
    return tag.rsplit("}", 1)[-1]


def normalized_color(asset: FigureAsset, value: str) -> str:
    key = value.strip().lower()
    if asset.master_name == "Figure_3.svg":
        gradient_color = remap_figure3_gradient_color(key)
        if gradient_color:
            return gradient_color
    return COLOR_MAPS[asset.master_name].get(key, COMMON_COLORS.get(key, value))


def normalize_css(asset: FigureAsset, css: str) -> str:
    css = FONT_DECLARATION_RE.sub(lambda match: match.group(1) + FONT_FAMILY, css)
    return PAINT_DECLARATION_RE.sub(
        lambda match: match.group(1) + normalized_color(asset, match.group(2)),
        css,
    )


def normalize_inline_style(asset: FigureAsset, style: str) -> str:
    return normalize_css(asset, style)


def apply_semantic_overrides(asset: FigureAsset, element: ET.Element) -> None:
    """Resolve source colors that represented two different concepts."""

    if asset.master_name != "Figure_1.svg":
        return
    class_name = element.attrib.get("class", "")
    if class_name == "st11":
        element.set("style", f"fill:{HARNESS_TINT};stroke:{HARNESS}")
    elif class_name == "st23":
        element.set("style", f"fill:{HARNESS}")


def parse_svg(data: bytes) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.ElementTree(ET.fromstring(data, parser=parser))


def _metadata_element(name: str, identifier: str, text: str) -> ET.Element:
    node = ET.Element(f"{{{SVG_NS}}}{name}", {"id": identifier})
    node.text = text
    return node


def _format_inches(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") + "in"


def normalize_svg(asset: FigureAsset) -> bytes:
    source_path = MASTER_DIR / asset.master_name
    source_data = source_path.read_bytes()
    actual_sha256 = hashlib.sha256(source_data).hexdigest()
    if actual_sha256 != asset.source_sha256:
        raise AssertionError(
            f"Physician master changed: {asset.master_name}: {actual_sha256}"
        )

    tree = parse_svg(source_data)
    root = tree.getroot()
    if local_name(root.tag) != "svg":
        raise AssertionError(f"Not an SVG root: {asset.master_name}")
    view_box = root.attrib.get("viewBox")
    if not view_box:
        raise AssertionError(f"Missing viewBox: {asset.master_name}")
    _, _, width, height = [float(item) for item in view_box.split()]
    root.set("width", "7.5in")
    root.set("height", _format_inches(7.5 * height / width))
    root.set("role", "img")
    root.set("aria-labelledby", "figure-title figure-description")

    for child in list(root):
        if local_name(child.tag) in {"title", "desc"}:
            root.remove(child)
    root.insert(
        0,
        _metadata_element("title", "figure-title", asset.title),
    )
    root.insert(
        1,
        _metadata_element(
            "desc", "figure-description", asset.description
        ),
    )

    for element in root.iter():
        name = local_name(element.tag)
        if name == "style" and element.text:
            element.text = normalize_css(asset, element.text)
        for attribute in ("fill", "stroke"):
            if attribute in element.attrib:
                element.set(attribute, normalized_color(asset, element.attrib[attribute]))
        if "font-family" in element.attrib:
            element.set("font-family", FONT_FAMILY)
        if "style" in element.attrib:
            element.set(
                "style", normalize_inline_style(asset, element.attrib["style"])
            )
        apply_semantic_overrides(asset, element)

    if geometry_dom_hash(source_data) != geometry_dom_hash_from_tree(tree):
        raise AssertionError(f"Normalization changed DOM or geometry: {asset.master_name}")

    ET.indent(tree, space="  ")
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    return buffer.getvalue()


def _sanitized_style(style: str) -> str:
    value = REMOVABLE_DECLARATION_RE.sub("", style)
    value = re.sub(r";+", ";", value)
    return re.sub(r"\s+", " ", value).strip(" ;")


def _invariant_node(element: ET.Element, *, is_root: bool = False) -> tuple:
    name = local_name(element.tag)
    if name in {"title", "desc"} and is_root:
        return ()
    attributes: list[tuple[str, str]] = []
    for key, value in element.attrib.items():
        key_name = local_name(key)
        if is_root and key_name in {"width", "height", "role", "aria-labelledby"}:
            continue
        if key_name in {"fill", "stroke", "font-family"}:
            continue
        if key_name == "style":
            sanitized = _sanitized_style(value)
            if sanitized:
                attributes.append((key, sanitized))
            continue
        attributes.append((key, value))
    if name == "style":
        text = _sanitized_style(element.text or "")
    elif name in {"text", "tspan"}:
        raw_text = element.text or ""
        text = raw_text if raw_text.strip() else ""
    else:
        raw_text = element.text or ""
        text = raw_text if raw_text.strip() else ""
    children = tuple(
        node
        for child in element
        if (node := _invariant_node(child, is_root=False))
    )
    return (name, tuple(attributes), text, children)


def geometry_dom_hash_from_tree(tree: ET.ElementTree) -> str:
    root = tree.getroot()
    children = tuple(
        node
        for child in root
        if local_name(child.tag) not in {"title", "desc"}
        if (node := _invariant_node(child, is_root=False))
    )
    root_tuple = (
        local_name(root.tag),
        tuple(
            (key, value)
            for key, value in root.attrib.items()
            if local_name(key) not in {"width", "height", "role", "aria-labelledby"}
        ),
        "",
        children,
    )
    return hashlib.sha256(repr(root_tuple).encode("utf-8")).hexdigest()


def geometry_dom_hash(data: bytes) -> str:
    return geometry_dom_hash_from_tree(parse_svg(data))


def render_png(svg_data: bytes, output_path: Path) -> tuple[int, int]:
    raw_png = cairosvg.svg2png(
        bytestring=svg_data,
        output_width=OUTPUT_WIDTH_PX,
        dpi=OUTPUT_DPI,
    )
    with Image.open(io.BytesIO(raw_png)) as image:
        image.load()
        output = io.BytesIO()
        image.save(
            output,
            format="PNG",
            dpi=(OUTPUT_DPI, OUTPUT_DPI),
            optimize=False,
            compress_level=6,
        )
        dimensions = image.size
    output_path.write_bytes(output.getvalue())
    if dimensions[0] != OUTPUT_WIDTH_PX:
        raise AssertionError(f"Unexpected PNG width: {output_path}: {dimensions}")
    if output_path.stat().st_size >= 5 * 1024 * 1024:
        raise AssertionError(f"PNG exceeds 5 MiB: {output_path}")
    return dimensions


def write_manifest(paths: list[Path]) -> None:
    lines = [f"{file_sha256(path)}  {path.name}" for path in paths]
    (OUTPUT_DIR / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    svg_paths: list[Path] = []
    png_paths: list[Path] = []
    for asset in ASSETS:
        svg_data = normalize_svg(asset)
        svg_path = OUTPUT_DIR / asset.svg_name
        mirror_path = MIRROR_DIR / asset.mirror_name
        png_path = OUTPUT_DIR / asset.png_name
        svg_path.write_bytes(svg_data)
        mirror_path.write_bytes(svg_data)
        render_png(svg_data, png_path)
        svg_paths.append(svg_path)
        png_paths.append(png_path)
    write_manifest(svg_paths + png_paths)
    print(
        f"Normalized six physician SVG masters and rendered six {OUTPUT_DPI} DPI PNGs in {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

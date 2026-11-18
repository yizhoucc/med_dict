#!/usr/bin/env python3
"""Check that v2.2.1 preserves the six physician SVG layouts exactly."""

from __future__ import annotations

import base64
import hashlib
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RATINGS = ROOT / "results/extraction_comparison/clinician_ratings"
FIGURES = RATINGS / "draft_figures"
MIRROR = ROOT / "manuscript/figures_svg"
DOWNLOADS = Path.home() / "Downloads"
sys.path.insert(0, str(RATINGS))

import physician_svg_normalizer as normalizer  # noqa: E402


DIMENSIONS = {
    "figure1_development_pathway_rough.svg": (2250, 1650),
    "figure2_evaluator_distribution_rough.svg": (2250, 1780),
    "figure3_interrater_matrix_rough.svg": (2250, 1140),
    "figure4_effect_profile.svg": (2250, 727),
    "supplementary_figure_s1_note_margins.svg": (2250, 540),
    "supplementary_figure_s2_staged_ablation.svg": (2250, 959),
}

DOWNLOAD_NAMES = {
    "Figure_1.svg": "Figure_1(1).svg",
    "Figure_2.svg": "Figure_2(1).svg",
    "Figure_3.svg": "Figure_3(1).svg",
    "Figure_4.svg": "Figure_4(2).svg",
    "Supplementary_Figure_S1.svg": "Supplementary_Figure_S1(1).svg",
    "Supplementary_Figure_S2.svg": "Supplementary_Figure_S2(1).svg",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_nodes(data: bytes) -> list[str]:
    root = normalizer.parse_svg(data).getroot()
    return [
        node.text or ""
        for node in root.iter()
        if normalizer.local_name(node.tag) in {"text", "tspan"}
        if (node.text or "").strip()
    ]


def assert_clear_border(path: Path) -> None:
    with Image.open(path) as image:
        pixels = image.convert("RGB").load()
        width, height = image.size
        border = (
            [pixels[x, 0] for x in range(width)]
            + [pixels[x, height - 1] for x in range(width)]
            + [pixels[0, y] for y in range(height)]
            + [pixels[width - 1, y] for y in range(height)]
        )
        assert all(pixel == (255, 255, 255) for pixel in border), path.name


def main() -> None:
    assert (
        normalizer.HARNESS,
        normalizer.BASELINE,
        normalizer.TIE,
        normalizer.PHYSICIAN,
        normalizer.LLM_REVIEW,
        normalizer.NEUTRAL,
        normalizer.INK,
        normalizer.GRID,
    ) == (
        "#2F6690",
        "#B85C45",
        "#C2C5C9",
        "#4C8872",
        "#766A91",
        "#788692",
        "#293741",
        "#E7EAEC",
    )
    assert (
        normalizer.HARNESS_TINT,
        normalizer.BASELINE_TINT,
        normalizer.PHYSICIAN_TINT,
        normalizer.LLM_TINT,
        normalizer.NEUTRAL_TINT,
        normalizer.PANEL_FILL,
    ) == ("#EAF0F4", "#F8EFEC", "#EDF3F1", "#F1F0F4", "#F2F3F4", "#FAFAFB")
    manifest = {
        name: digest
        for digest, name in (
            line.split(None, 1)
            for line in (FIGURES / "MANIFEST.sha256").read_text().splitlines()
            if line.strip()
        )
    }
    output_svgs: list[bytes] = []
    namespace = "{http://www.w3.org/2000/svg}"
    for asset in normalizer.ASSETS:
        master_path = normalizer.MASTER_DIR / asset.master_name
        master_data = master_path.read_bytes()
        assert sha256(master_data) == asset.source_sha256
        download = DOWNLOADS / DOWNLOAD_NAMES[asset.master_name]
        if download.exists():
            assert download.read_bytes() == master_data

        svg_path = FIGURES / asset.svg_name
        png_path = FIGURES / asset.png_name
        svg_data = svg_path.read_bytes()
        output_svgs.append(svg_data)
        assert sha256(svg_data) == manifest[svg_path.name]
        assert (MIRROR / asset.mirror_name).read_bytes() == svg_data
        assert normalizer.geometry_dom_hash(master_data) == normalizer.geometry_dom_hash(svg_data)
        assert text_nodes(master_data) == text_nodes(svg_data)

        master_root = normalizer.parse_svg(master_data).getroot()
        root = normalizer.parse_svg(svg_data).getroot()
        assert root.attrib["viewBox"] == master_root.attrib["viewBox"]
        _, _, width, height = [float(item) for item in root.attrib["viewBox"].split()]
        assert root.attrib["width"] == "7.5in"
        assert root.attrib["height"] == normalizer._format_inches(7.5 * height / width)
        assert root.attrib["role"] == "img"
        assert root.attrib["aria-labelledby"] == "figure-title figure-description"
        assert root.find(f"{namespace}title") is not None
        assert root.find(f"{namespace}desc") is not None
        assert root.findall(f".//{namespace}text")
        svg_source = svg_data.decode("utf-8")
        assert normalizer.FONT_FAMILY in svg_source
        assert "ArialMT" not in svg_source and "Arial-BoldMT" not in svg_source
        if asset.master_name != "Figure_3.svg":
            allowed_colors = {
                value.upper()
                for value in [
                    normalizer.HARNESS,
                    normalizer.BASELINE,
                    normalizer.TIE,
                    normalizer.PHYSICIAN,
                    normalizer.LLM_REVIEW,
                    normalizer.NEUTRAL,
                    normalizer.INK,
                    normalizer.GRID,
                    normalizer.PANEL_FILL,
                    normalizer.WHITE,
                    normalizer.HARNESS_TINT,
                    normalizer.BASELINE_TINT,
                    normalizer.PHYSICIAN_TINT,
                    normalizer.LLM_TINT,
                    normalizer.NEUTRAL_TINT,
                ]
            }
            observed_colors = {
                value.upper() for value in re.findall(r"#[0-9a-fA-F]{6}", svg_source)
            }
            assert observed_colors <= allowed_colors, (asset.svg_name, observed_colors - allowed_colors)

        with Image.open(png_path) as image:
            assert image.size == DIMENSIONS[asset.svg_name]
            dpi = image.info.get("dpi")
            assert dpi and all(abs(value - 300) < 0.01 for value in dpi)
        assert sha256(png_path.read_bytes()) == manifest[png_path.name]
        assert_clear_border(png_path)

    f1 = normalizer.parse_svg(output_svgs[0]).getroot()
    f1_text = " ".join(text_nodes(output_svgs[0]))
    assert all(f"{panel}" in f1_text for panel in "ABCD") and "Panel E" not in f1_text
    master_f1 = normalizer.parse_svg((normalizer.MASTER_DIR / "Figure_1.svg").read_bytes()).getroot()
    assert sum(1 for _ in f1.iter()) == sum(1 for _ in master_f1.iter()) + 2

    for index in (1, 3, 5):
        root = ET.fromstring(output_svgs[index])
        assert not root.findall(f".//{namespace}pattern")
        assert "hatch" not in output_svgs[index].decode("utf-8").lower()
    f2_root = ET.fromstring(output_svgs[1])
    assert not f2_root.findall(f".//{namespace}polygon")
    assert not f2_root.findall(f".//{namespace}path")
    assert len(f2_root.findall(f".//{namespace}circle")) == 8
    assert "diamond" not in output_svgs[1].decode("utf-8").lower()
    assert "lollipop" not in output_svgs[1].decode("utf-8").lower()

    f3_root = ET.fromstring(output_svgs[2])
    heatmap_scale = [
        node.attrib["fill"].upper()
        for node in f3_root.findall(f".//{namespace}rect")
        if node.attrib.get("y") == "715"
    ]
    assert len(heatmap_scale) == 100
    assert heatmap_scale == [
        normalizer.interpolate_color(
            normalizer.HEATMAP_LIGHT,
            normalizer.HEATMAP_DARK,
            index / 99,
        )
        for index in range(100)
    ]

    combined = b"\n".join(output_svgs).decode("utf-8").upper()
    for color in [
        normalizer.HARNESS,
        normalizer.BASELINE,
        normalizer.TIE,
        normalizer.PHYSICIAN,
        normalizer.LLM_REVIEW,
        normalizer.NEUTRAL,
        normalizer.INK,
        normalizer.GRID,
    ]:
        assert color in combined, color
    for rejected in ["#0072B2", "#D55E00", "#B8B8B8", "#009E73", "#7A5FA8", "#718096", "#243342", "#E6E6E6"]:
        assert rejected not in combined, rejected

    document = (ROOT / "WORKSHOP_PAPER_DRAFT.html").read_text()
    embedded = re.findall(
        r'<img class="figure-screen" src="data:image/svg\+xml;base64,([^"]+)"',
        document,
    )
    assert len(embedded) == 6
    assert [base64.b64decode(item) for item in embedded] == output_svgs
    print(
        "PHYSICIAN_SVG_NORMALIZATION_OK masters=6 geometry=exact palette=clinical "
        "svg=6 png=6 width=2250 dpi=300 embedded=6 borders=clear"
    )


if __name__ == "__main__":
    main()

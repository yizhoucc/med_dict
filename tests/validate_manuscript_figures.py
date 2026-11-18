#!/usr/bin/env python3
"""Validate the v2.1.1 manuscript figure assets and embedded copies."""

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
sys.path.insert(0, str(RATINGS))

import manuscript_figure_renderer as renderer  # noqa: E402


ASSETS = {
    "figure1_development_pathway_rough": ("Figure_1.svg", (2250, 1800), "7.5in", "6in"),
    "figure2_evaluator_distribution_rough": ("Figure_2.svg", (2250, 1650), "7.5in", "5.5in"),
    "figure3_interrater_matrix_rough": ("Figure_3.svg", (2250, 1140), "7.5in", "3.8in"),
    "figure4_effect_profile": ("Figure_4.svg", (2250, 900), "7.5in", "3in"),
    "supplementary_figure_s1_note_margins": ("Supplementary_Figure_S1.svg", (2250, 540), "7.5in", "1.8in"),
    "supplementary_figure_s2_staged_ablation": ("Supplementary_Figure_S2.svg", (2250, 1020), "7.5in", "3.4in"),
}


def main() -> None:
    renderer.assert_data(renderer.load_raters())
    manifest = {
        name: digest
        for digest, name in (
            line.split(None, 1)
            for line in (FIGURES / "MANIFEST.sha256").read_text().splitlines()
            if line.strip()
        )
    }
    svg_bytes: list[bytes] = []
    namespace = "{http://www.w3.org/2000/svg}"
    for stem, (mirror_name, dimensions, width, height) in ASSETS.items():
        svg_path = FIGURES / f"{stem}.svg"
        png_path = FIGURES / f'{stem.replace("_rough", "")}_print.png'
        svg_data = svg_path.read_bytes()
        svg_bytes.append(svg_data)
        assert hashlib.sha256(svg_data).hexdigest() == manifest[svg_path.name]
        assert (MIRROR / mirror_name).read_bytes() == svg_data
        root = ET.fromstring(svg_data)
        assert root.attrib["width"] == width and root.attrib["height"] == height
        assert root.attrib["viewBox"] and root.attrib["role"] == "img"
        assert root.attrib["aria-labelledby"] == "figure-title figure-description"
        title = root.find(f"{namespace}title")
        desc = root.find(f"{namespace}desc")
        assert title is not None and title.attrib.get("id") == "figure-title" and title.text
        assert desc is not None and desc.attrib.get("id") == "figure-description" and desc.text
        assert root.findall(f".//{namespace}text"), f"Text is not editable in {svg_path.name}"
        svg_text = svg_data.decode("utf-8")
        for font in ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]:
            assert font in svg_text
        image = Image.open(png_path)
        assert image.size == dimensions
        dpi = image.info.get("dpi")
        assert dpi and all(abs(value - 300) < 0.01 for value in dpi)
        assert hashlib.sha256(png_path.read_bytes()).hexdigest() == manifest[png_path.name]

    combined = b"\n".join(svg_bytes).decode("utf-8")
    for color in [renderer.HARNESS, renderer.BASELINE, renderer.TIE, renderer.PHYSICIAN, renderer.LLM_REVIEW, renderer.TEXT, renderer.GRID, renderer.PANEL_FILL]:
        assert color.lower() in combined.lower(), color
    assert "hatch" not in combined.lower() or "pattern" in combined.lower()

    document = (ROOT / "WORKSHOP_PAPER_DRAFT.html").read_text()
    embedded = re.findall(r'<img class="figure-screen" src="data:image/svg\+xml;base64,([^"]+)"', document)
    assert len(embedded) == 6
    decoded = [base64.b64decode(item) for item in embedded]
    assert decoded == svg_bytes
    print("FIGURE_VALIDATION_OK svg=6 png=6 width=2250 dpi=300 embedded=6")


if __name__ == "__main__":
    main()

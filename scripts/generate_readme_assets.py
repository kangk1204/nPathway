#!/usr/bin/env python3
"""Generate GitHub README preview assets from current dashboard figures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"

BULK_FIG = ROOT / "results" / "demo_bulk_case_vs_control_20260308_v4" / "figures"
AD_FIG = ROOT / "results" / "ad_bulk_GSE203206_ensemble" / "dashboard" / "figures"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Avenir Next.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_cover(image: Image.Image, width: int, height: int) -> Image.Image:
    src = image.copy().convert("RGB")
    scale = max(width / src.width, height / src.height)
    resized = src.resize((int(src.width * scale), int(src.height * scale)))
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _card(base: Image.Image, xy: tuple[int, int], size: tuple[int, int], fill: str = "#fffdf8") -> ImageDraw.ImageDraw:
    draw = ImageDraw.Draw(base)
    x, y = xy
    w, h = size
    draw.rounded_rectangle((x, y, x + w, y + h), radius=28, fill=fill, outline="#d8d8d8", width=1)
    return draw


def _paste_card_image(base: Image.Image, src_path: Path, xy: tuple[int, int], size: tuple[int, int]) -> None:
    image = Image.open(src_path)
    fitted = _fit_cover(image, *size)
    x, y = xy
    mask = Image.new("L", size, 255)
    base.paste(fitted, (x, y), mask)


def generate_dashboard_preview() -> None:
    canvas = Image.new("RGB", (1700, 1180), "#f6efe4")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(48, bold=True)
    subtitle_font = _font(22)
    label_font = _font(18, bold=True)
    small_font = _font(16)

    draw.rounded_rectangle((40, 34, 1660, 1140), radius=36, fill="#fffaf1", outline="#d9d0c6", width=2)
    draw.text((86, 74), "nPathway Dashboard Preview", fill="#14213d", font=title_font)
    draw.text(
        (88, 134),
        "New default layout: explorer first, enrichment second, multi-pathway curves at the top, and dense tables grouped into tabs.",
        fill="#566272",
        font=subtitle_font,
    )

    pills = [
        "Program Explorer",
        "Enriched Programs",
        "Multi-Pathway Enrichment Curves",
        "Interactive Tables",
    ]
    x = 88
    for pill in pills:
        w = draw.textlength(pill, font=small_font) + 34
        draw.rounded_rectangle((x, 186, x + w, 222), radius=18, fill="#eaf2fb", outline="#bfd0e6")
        draw.text((x + 16, 195), pill, fill="#204c83", font=small_font)
        x += int(w) + 12

    # Left: Enriched Programs
    _card(canvas, (84, 254), (980, 390))
    draw.text((112, 278), "Enriched Programs", fill="#173153", font=label_font)
    draw.text((112, 304), "Top signal is visible before the user scrolls.", fill="#667281", font=small_font)
    _paste_card_image(canvas, BULK_FIG / "figure_3_claim_gates.png", (104, 336), (430, 280))
    _paste_card_image(canvas, BULK_FIG / "figure_2_program_sizes.png", (552, 336), (484, 280))

    # Right: Spotlight + table tabs
    _card(canvas, (1092, 254), (520, 390))
    draw.text((1120, 278), "Study Summary Hub", fill="#173153", font=label_font)
    draw.text((1120, 304), "Narrative, downloads, metrics, and claim gates share one card.", fill="#667281", font=small_font)
    tab_x = 1120
    for idx, pill in enumerate(["Checklist", "Downloads", "Headline", "Claim gates"]):
        fill = "#eaf2fb" if idx == 0 else "#f7f1e8"
        draw.rounded_rectangle((tab_x, 344, tab_x + 108, 378), radius=16, fill=fill, outline="#d5cabc")
        draw.text((tab_x + 14, 353), pill, fill="#23405f", font=small_font)
        tab_x += 116
    draw.rounded_rectangle((1120, 400, 1588, 602), radius=22, fill="#fffdf9", outline="#e0d7cd")
    draw.text((1146, 426), "Reviewer Checklist", fill="#173153", font=label_font)
    rows = [
        "Lead enrichment",
        "Reference anchor",
        "Strongest context shift",
        "Claim-supported programs",
    ]
    for i, row in enumerate(rows):
        yy = 470 + i * 32
        draw.rounded_rectangle((1146, yy, 1560, yy + 24), radius=12, fill="#f3eee5")
        draw.text((1160, yy + 4), row, fill="#5e6876", font=small_font)

    # Bottom full-width: multi-pathway curves
    _card(canvas, (84, 676), (1528, 388))
    draw.text((112, 700), "Multi-Pathway Enrichment Curves", fill="#173153", font=label_font)
    draw.text((112, 726), "Classic running-sum curves and leading-edge preview live above the fold.", fill="#667281", font=small_font)
    _paste_card_image(canvas, BULK_FIG / "figure_6_multi_pathway_enrichment_curves.png", (104, 760), (920, 276))
    draw.rounded_rectangle((1046, 760, 1584, 1036), radius=22, fill="#fffdf9", outline="#e0d7cd")
    draw.text((1072, 784), "Interactive Tables", fill="#173153", font=label_font)
    tab_x = 1072
    for idx, pill in enumerate(["Program Summary", "Context Drivers", "Core Genes"]):
        fill = "#eaf2fb" if idx == 0 else "#f7f1e8"
        w = draw.textlength(pill, font=small_font) + 28
        draw.rounded_rectangle((tab_x, 822, tab_x + w, 856), radius=16, fill=fill, outline="#d5cabc")
        draw.text((tab_x + 14, 831), pill, fill="#23405f", font=small_font)
        tab_x += int(w) + 10
    for i in range(7):
        yy = 888 + i * 20
        draw.rounded_rectangle((1072, yy, 1558, yy + 14), radius=7, fill="#f3eee5")

    canvas.save(ASSET_DIR / "npathway_dashboard_preview.png")


def generate_workflow_overview() -> None:
    canvas = Image.new("RGB", (1600, 950), "#f5efe4")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    subtitle_font = _font(20)
    box_title_font = _font(24, bold=True)
    body_font = _font(18)

    draw.text((78, 56), "nPathway Workflow", fill="#14213d", font=title_font)
    draw.text(
        (80, 112),
        "From matrix and metadata to dynamic programs, pathway alignment, and a browser-ready dashboard.",
        fill="#5b6777",
        font=subtitle_font,
    )

    stages = [
        ("1. Input", "Expression matrix + metadata\nBulk or scRNA pseudobulk", (78, 200), "#fffaf3"),
        ("2. Discovery", "Dynamic gene programs\nContext-aware evidence", (560, 200), "#fffaf3"),
        ("3. Review", "Curated pathway matching\nInteractive HTML dashboard", (1042, 200), "#fffaf3"),
    ]
    for title, body, xy, fill in stages:
        x, y = xy
        draw.rounded_rectangle((x, y, x + 420, y + 180), radius=28, fill=fill, outline="#d8d0c6", width=2)
        draw.text((x + 26, y + 24), title, fill="#173153", font=box_title_font)
        for idx, line in enumerate(body.splitlines()):
            draw.text((x + 26, y + 78 + idx * 28), line, fill="#5f6b79", font=body_font)
    draw.line((500, 290, 540, 290), fill="#2a9d8f", width=6)
    draw.line((982, 290, 1022, 290), fill="#2a9d8f", width=6)

    _card(canvas, (78, 444), (450, 414))
    draw.text((106, 466), "Differential signal", fill="#173153", font=box_title_font)
    _paste_card_image(canvas, BULK_FIG / "figure_1_volcano.png", (100, 514), (406, 320))

    _card(canvas, (572, 444), (450, 414))
    draw.text((600, 466), "Program landscape", fill="#173153", font=box_title_font)
    _paste_card_image(canvas, BULK_FIG / "figure_2_program_sizes.png", (594, 514), (406, 320))

    _card(canvas, (1066, 444), (450, 414))
    draw.text((1094, 466), "Enrichment review", fill="#173153", font=box_title_font)
    _paste_card_image(canvas, AD_FIG / "figure_6_multi_pathway_enrichment_curves.png", (1088, 514), (406, 320))

    canvas.save(ASSET_DIR / "npathway_workflow_overview.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generate_workflow_overview()
    generate_dashboard_preview()
    print("Generated README assets:")
    print("-", ASSET_DIR / "npathway_workflow_overview.png")
    print("-", ASSET_DIR / "npathway_dashboard_preview.png")


if __name__ == "__main__":
    main()

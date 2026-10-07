"""Применяет светлую футуристичную тему к презентации проекта."""

from pathlib import Path
from shutil import copy2

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt


SOURCE = Path("Local_RAG_Project_Presentation.pptx")
BACKUP = Path("Local_RAG_Project_Presentation_dark_backup.pptx")

BACKGROUND = "09182D"
INK = "F2F7FF"
MUTED = "AFC0D5"
CARD = "132A48"
CARD_ALT = "1B385D"
GRID = "294A70"
CYAN = "35D7F2"
BLUE = "7890FF"
GREEN = "38DDB1"
AMBER = "F7C65B"
CORAL = "FF7A90"

FILL_MAP = {
    "142033": CARD,
    "1B2B42": CARD_ALT,
    "22D3EE": CYAN,
    "3B82F6": BLUE,
    "34D399": GREEN,
    "FBBF24": AMBER,
    "F87171": CORAL,
}

LINE_MAP = {
    "142033": GRID,
    "1B2B42": "35577E",
    "22D3EE": CYAN,
    "3B82F6": BLUE,
    "34D399": GREEN,
    "FBBF24": AMBER,
    "F87171": CORAL,
}

TEXT_MAP = {
    "F1F5F9": INK,
    "9BABC2": MUTED,
    "22D3EE": CYAN,
    "3B82F6": BLUE,
    "34D399": GREEN,
    "FBBF24": AMBER,
    "F87171": CORAL,
}


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def current_rgb(color) -> str | None:
    try:
        value = color.rgb
        return str(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def add_future_grid(slide, slide_width, slide_height) -> None:
    """Добавляет ненавязчивые технологичные маркеры по краям слайда."""
    top_line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(8.95),
        Inches(0.24),
        Inches(3.65),
        Pt(1.2),
    )
    top_line.fill.solid()
    top_line.fill.fore_color.rgb = rgb("35577E")
    top_line.line.fill.background()

    accent_line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(11.75),
        Inches(0.235),
        Inches(0.85),
        Pt(2.4),
    )
    accent_line.fill.solid()
    accent_line.fill.fore_color.rgb = rgb(CYAN)
    accent_line.line.fill.background()

    for x, y, size, color in (
        (12.40, 6.72, 0.10, CYAN),
        (12.18, 6.72, 0.06, BLUE),
        (0.24, 6.74, 0.06, GREEN),
    ):
        dot = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(size), Inches(size)
        )
        dot.fill.solid()
        dot.fill.fore_color.rgb = rgb(color)
        dot.line.fill.background()


def restyle(source: Path = SOURCE, output: Path = SOURCE) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source == output and not BACKUP.exists():
        copy2(source, BACKUP)

    prs = Presentation(source)
    for slide in prs.slides:
        background = slide.background.fill
        background.solid()
        background.fore_color.rgb = rgb(BACKGROUND)

        for shape in list(slide.shapes):
            if getattr(shape, "fill", None) and shape.fill.type:
                try:
                    old = current_rgb(shape.fill.fore_color)
                except TypeError:
                    old = None
                if old in FILL_MAP:
                    shape.fill.fore_color.rgb = rgb(FILL_MAP[old])

            # Скругляем карточки, плашки и акцентные полосы. Круги и другие
            # смысловые фигуры сохраняют исходную геометрию.
            try:
                preset_geometry = shape._sp.spPr.prstGeom
                if (
                    preset_geometry is not None
                    and preset_geometry.get("prst") == "rect"
                    and shape.height > Inches(0.08)
                ):
                    preset_geometry.set("prst", "roundRect")
            except AttributeError:
                pass

            if getattr(shape, "line", None):
                old = current_rgb(shape.line.color)
                if old in LINE_MAP:
                    shape.line.color.rgb = rgb(LINE_MAP[old])
                    if old in {"142033", "1B2B42"}:
                        shape.line.width = Pt(1)

            if getattr(shape, "has_text_frame", False):
                text_frame = shape.text_frame
                text_frame.word_wrap = True
                text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
                text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
                text_frame.margin_left = Inches(0.08)
                text_frame.margin_right = Inches(0.08)
                text_frame.margin_top = Inches(0.04)
                text_frame.margin_bottom = Inches(0.04)
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.color.rgb = rgb("FFFFFF")
                        size = run.font.size.pt if run.font.size else 0
                        run.font.name = "Aptos Display" if size >= 24 else "Aptos"
                        if size >= 30:
                            run.font.bold = True

        add_future_grid(slide, prs.slide_width, prs.slide_height)

    prs.save(output)


if __name__ == "__main__":
    restyle(source=BACKUP if BACKUP.exists() else SOURCE, output=SOURCE)
    print(f"Updated: {SOURCE}")
    print(f"Backup:  {BACKUP}")

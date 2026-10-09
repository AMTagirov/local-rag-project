import re
import os
import logging
import unicodedata
from pathlib import Path
from typing import Generator

import numpy as np
import pymupdf

from src.config.schema import DocumentAnalysisConfig


logger = logging.getLogger(__name__)

MATH_SYMBOLS = frozenset("∑∫√∞≈≠≤≥±×÷∂∇∆∏∈∉⊂⊆∪∩→↔∀∃αβγδεζηθλμπρστφψω")
MATH_TEXT_PATTERNS = (
    re.compile(r"\b(?:sin|cos|tan|log|ln|lim|exp)\s*\(", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]\s*[=≈<>≤≥]\s*[-+]?\d"),
    re.compile(r"\b\w+_[A-Za-z0-9]+\b"),
    re.compile(r"\b\w+\^[A-Za-z0-9]+\b"),
)


def normalize_structured_markdown(text: str) -> str:
    """
    Мягко очищает Markdown, сохраняя:
    - строки таблиц;
    - блоки LaTeX;
    - заголовки;
    - ссылки на изображения.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Удаляем управляющие символы, сохраняя переносы и табуляцию.
    text = "".join(
        char
        for char in text
        if char in "\n\t"
        or unicodedata.category(char) not in {"Cc", "Cf"}
    )

    # Убираем пробелы в конце строк.
    lines = [line.rstrip() for line in text.splitlines()]

    # Не допускаем больше двух пустых строк подряд.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class PaddleDocumentAnalyzer:
    """
    Адаптер PP-StructureV3.

    Модель загружается только при первом разборе PDF.
    Поэтому обычный запуск RAG по уже созданной коллекции
    не расходует память на OCR-модели.
    """

    def __init__(self, config: DocumentAnalysisConfig):
        self.config = config
        self._pipeline = None
        self._formula_recognizer = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            if (
                self.config.engine == "onnxruntime"
                and self.config.use_formula_recognition
            ):
                raise RuntimeError(
                    "Официальный PP-FormulaNet не поставляется в ONNX-формате. "
                    "Для формул используйте engine='paddle_dynamic' и "
                    "formula_model_name='PP-FormulaNet_plus-L'; этому профилю "
                    "нужно больше оперативной памяти или GPU."
                )

            # PaddleX читает путь к кэшу во время импорта. Храним модели внутри
            # data/, чтобы запуск не зависел от доступности домашнего каталога.
            cache_dir = Path(self.config.cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
            os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))

            # WSL предоставляет драйвер CUDA через этот каталог. Paddle может
            # видеть GPU, но не находить libcuda.so без явного пути.
            wsl_cuda_dir = Path("/usr/lib/wsl/lib")
            if self.config.device.startswith("gpu") and wsl_cuda_dir.exists():
                library_path = os.environ.get("LD_LIBRARY_PATH", "")
                path_parts = [part for part in library_path.split(os.pathsep) if part]
                if str(wsl_cuda_dir) not in path_parts:
                    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                        [str(wsl_cuda_dir), *path_parts]
                    )
            try:
                from paddleocr import PPStructureV3
            except ImportError as error:
                raise RuntimeError(
                    "Document analysis включён, но PaddleOCR не установлен. "
                    "Установите requirements-ocr.txt."
                ) from error

            try:
                self._pipeline = PPStructureV3(
                    device=self.config.device,
                    engine=self.config.engine,
                    lang=self.config.ocr_language,
                    ocr_version=self.config.ocr_version,
                    # Для обычных цифровых PDF эти три модели не нужны. Их
                    # отключение заметно снижает расход памяти на CPU.
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    use_seal_recognition=False,
                    formula_recognition_model_name=(
                        self.config.formula_model_name
                    ),
                    use_formula_recognition=(
                        self.config.use_formula_recognition
                        and not self.config.formula_filter_enabled
                    ),
                    use_table_recognition=(
                        self.config.use_table_recognition
                        and not self.config.prefer_native_pdf_text
                    ),
                    use_chart_recognition=(
                        self.config.use_chart_recognition
                    ),
                )
            except RuntimeError as error:
                if "dependency error" in str(error).lower():
                    raise RuntimeError(
                        "Для PP-StructureV3 не установлены дополнительные "
                        "зависимости PaddleOCR. Выполните: "
                        "venv/bin/python -m pip install -r requirements-ocr.txt"
                    ) from error
                raise

        return self._pipeline

    @property
    def formula_recognizer(self):
        if self._formula_recognizer is None:
            from paddleocr import FormulaRecognition

            self._formula_recognizer = FormulaRecognition(
                model_name=self.config.formula_model_name,
                device=self.config.device,
                engine=self.config.engine,
            )
        return self._formula_recognizer

    def parse_pdf(self, file_path: str) -> Generator[str, None, None]:
        source_path = Path(file_path)

        document_assets_dir = (
            Path(self.config.assets_dir)
            / source_path.stem
        )

        if self.config.save_images:
            document_assets_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        filter_formulas = (
            self.config.use_formula_recognition
            and self.config.formula_filter_enabled
        )

        results = self.pipeline.predict(
            input=str(source_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_formula_recognition=(
                self.config.use_formula_recognition and not filter_formulas
            ),
            use_table_recognition=(
                self.config.use_table_recognition
                and not self.config.prefer_native_pdf_text
            ),
            use_chart_recognition=(
                self.config.use_chart_recognition
            ),
            format_block_content=True,
        )

        has_content = False

        needs_pdf_document = (
            filter_formulas or self.config.prefer_native_pdf_text
        )
        pdf_document = (
            pymupdf.open(source_path) if needs_pdf_document else None
        )

        for page_index, result in enumerate(results):
            page = None
            formula_blocks = []
            if pdf_document is not None and page_index < len(pdf_document):
                page = pdf_document[page_index]
                if filter_formulas:
                    heuristic_score = self._formula_heuristic_score(page)
                    layout_score = 2 if self._layout_has_formula(result) else 0
                    combined_score = min(4, heuristic_score + layout_score)

                    if combined_score >= self.config.formula_filter_threshold:
                        logger.info(
                            "Страница %s отправлена в FormulaNet: score=%s",
                            page_index + 1,
                            combined_score,
                        )
                        formula_blocks = self._recognize_formula_regions(
                            page,
                            result,
                        )
                    else:
                        logger.info(
                            "Страница %s пропущена FormulaNet: score=%s",
                            page_index + 1,
                            combined_score,
                        )

            markdown_info = result.markdown

            markdown_text = markdown_info.get(
                "markdown_texts",
                "",
            )

            # Некоторые версии могут вернуть список частей.
            if isinstance(markdown_text, list):
                markdown_text = "\n\n".join(
                    str(part) for part in markdown_text
                )

            native_text = ""
            native_tables = ""
            if page is not None and self.config.prefer_native_pdf_text:
                tables = self._find_native_tables(page)
                table_rects = [pymupdf.Rect(table.bbox) for table in tables]
                native_text = self._extract_native_text(page, table_rects)
                if self.config.use_table_recognition:
                    native_tables = self._format_native_tables(tables)
                if (
                    len(native_text) < self.config.native_text_min_chars
                    and not native_tables
                ):
                    native_text = ""

            if native_text:
                markdown_text = native_text

            if native_tables:
                markdown_text = self._append_section(
                    markdown_text,
                    "Таблицы страницы",
                    native_tables,
                )

            if formula_blocks:
                markdown_text = self._append_formula_blocks(
                    markdown_text,
                    formula_blocks,
                )

            if self.config.save_images:
                if native_text and page is not None:
                    if not formula_blocks:
                        markdown_text = self._save_native_page_images(
                            markdown_text=markdown_text,
                            page=page,
                            assets_dir=document_assets_dir,
                            page_index=page_index,
                        )
                else:
                    markdown_text = self._save_page_images(
                        markdown_text=markdown_text,
                        images=markdown_info.get(
                            "markdown_images",
                            {},
                        ),
                        assets_dir=document_assets_dir,
                        page_index=page_index,
                    )

            markdown_text = normalize_structured_markdown(
                markdown_text
            )

            if markdown_text:
                has_content = True
                yield f"[Страница {page_index + 1}]\n\n{markdown_text}"

        if pdf_document is not None:
            pdf_document.close()

        if not has_content:
            raise RuntimeError(
                f"PP-StructureV3 не извлёк содержимое из "
                f"'{file_path}'"
            )

    def _recognize_formula_regions(self, page, layout_result) -> list[str]:
        scale = self.config.formula_render_dpi / 72
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        ).copy()
        crops = self._crop_formula_regions(image, layout_result)
        inputs = crops or [image]
        formula_blocks = []

        # PaddleX некорректно объединяет список разноразмерных ndarray в одно
        # широкое изображение, поэтому области обрабатываются последовательно.
        for formula_image in inputs:
            results = self.formula_recognizer.predict(input=formula_image)
            for result in results:
                formula = str(result.get("rec_formula", "")).strip()
                if formula:
                    formula = self._normalize_formula_latex(formula)
                    formula_blocks.append(f"$${formula}$$")

        if not formula_blocks:
            logger.warning("FormulaNet не распознал формулы на выбранной странице")
        return list(dict.fromkeys(formula_blocks))

    @staticmethod
    def _normalize_formula_latex(formula: str) -> str:
        formula = formula.strip().strip("$").strip()
        formula = re.sub(r"_\{\}", "", formula)
        return formula.rstrip(" ,.;")

    @staticmethod
    def _crop_formula_regions(image: np.ndarray, result) -> list[np.ndarray]:
        try:
            layout_result = result["layout_det_res"]
            layout_image = layout_result["input_img"]
            boxes = layout_result.get("boxes", [])
        except (KeyError, TypeError, AttributeError):
            return []

        source_height, source_width = layout_image.shape[:2]
        target_height, target_width = image.shape[:2]
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        formula_labels = {"formula", "inline_formula", "display_formula"}
        positioned_crops = []

        for box in boxes:
            if str(box.get("label", "")).lower() not in formula_labels:
                continue
            coordinates = box.get("coordinate", [])
            if len(coordinates) != 4:
                continue
            x_min, y_min, x_max, y_max = coordinates
            padding = 16
            left = max(0, int(x_min * scale_x) - padding)
            top = max(0, int(y_min * scale_y) - padding)
            right = min(target_width, int(x_max * scale_x) + padding)
            bottom = min(target_height, int(y_max * scale_y) + padding)
            if right > left and bottom > top:
                positioned_crops.append(
                    (top, left, image[top:bottom, left:right].copy())
                )

        positioned_crops.sort(key=lambda item: (item[0], item[1]))
        return [crop for _, _, crop in positioned_crops]

    @staticmethod
    def _extract_native_text(page, excluded_rects=None) -> str:
        excluded_rects = excluded_rects or []
        paragraphs = []
        for block in page.get_text("blocks", sort=True):
            if len(block) > 6 and block[6] != 0:
                continue
            block_rect = pymupdf.Rect(block[:4])
            center = pymupdf.Point(
                (block_rect.x0 + block_rect.x1) / 2,
                (block_rect.y0 + block_rect.y1) / 2,
            )
            if any(rect.contains(center) for rect in excluded_rects):
                continue
            text = str(block[4]).strip()
            if text:
                paragraphs.append(text)
        return normalize_structured_markdown("\n\n".join(paragraphs))

    @staticmethod
    def _find_native_tables(page) -> list:
        """Находит таблицы в цифровом PDF без запуска OCR-моделей."""
        try:
            return list(page.find_tables().tables)
        except Exception as error:
            logger.warning(
                "Не удалось извлечь таблицы со страницы %s: %s",
                page.number + 1,
                error,
            )
            return []

    @staticmethod
    def _format_native_tables(tables: list) -> str:
        markdown_tables = []
        for table in tables:
            rows = table.extract()
            if not rows:
                continue
            width = max(len(row) for row in rows)
            normalized_rows = []
            for row in rows:
                cells = list(row) + [""] * (width - len(row))
                normalized_rows.append([
                    str(cell or "")
                    .replace("\n", " ")
                    .replace("|", "\\|")
                    .strip()
                    for cell in cells
                ])
            header = normalized_rows[0]
            body = normalized_rows[1:]
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in header) + " |",
            ]
            lines.extend(
                "| " + " | ".join(row) + " |" for row in body
            )
            markdown_tables.append("\n".join(lines))
        return "\n\n".join(markdown_tables)

    @staticmethod
    def _append_section(text: str, title: str, content: str) -> str:
        if not text:
            return f"### {title}\n\n{content}"
        return f"{text}\n\n### {title}\n\n{content}"

    @staticmethod
    def _extract_formula_blocks(markdown_text: str) -> list[str]:
        if not markdown_text:
            return []
        blocks = re.findall(r"\$\$.+?\$\$", markdown_text, flags=re.DOTALL)
        if blocks:
            return [normalize_structured_markdown(block) for block in blocks]
        inline = re.findall(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", markdown_text)
        return [normalize_structured_markdown(block) for block in inline]

    @staticmethod
    def _append_formula_blocks(text: str, formula_blocks: list[str]) -> str:
        formula_section = "\n\n".join(formula_blocks)
        return PaddleDocumentAnalyzer._append_section(
            text,
            "Формулы страницы",
            formula_section,
        )

    @staticmethod
    def _formula_heuristic_score(page) -> int:
        text = page.get_text("text") or ""
        score = 0

        math_symbol_count = sum(character in MATH_SYMBOLS for character in text)
        if math_symbol_count >= 2 or any(
            pattern.search(text) for pattern in MATH_TEXT_PATTERNS
        ):
            score += 1

        operator_count = sum(text.count(operator) for operator in ("=", "+", "−", "*", "/"))
        digit_count = sum(character.isdigit() for character in text)
        if operator_count >= 4 and digit_count >= 8:
            score += 1

        has_images = bool(page.get_images(full=True))
        has_vector_graphics = len(page.get_drawings()) >= 5
        if has_images or has_vector_graphics:
            score += 1

        return min(score, 2)

    @staticmethod
    def _layout_has_formula(result) -> bool:
        try:
            layout_result = result["layout_det_res"]
            boxes = layout_result.get("boxes", [])
        except (KeyError, TypeError, AttributeError):
            return False

        formula_labels = {"formula", "inline_formula", "display_formula"}
        for box in boxes:
            label = str(
                box.get("label", box.get("cls_name", ""))
            ).lower()
            if label in formula_labels:
                return True
        return False

    @staticmethod
    def _save_page_images(
        markdown_text: str,
        images: dict,
        assets_dir: Path,
        page_index: int,
        append_missing: bool = False,
    ) -> str:
        """
        Сохраняет изображения из Markdown на диск и заменяет
        внутренние ссылки PaddleOCR на локальные пути.
        """
        saved_paths = []
        for image_number, (source_name, image) in enumerate(
            images.items(),
            start=1,
        ):
            output_name = (
                f"page_{page_index + 1:04d}_"
                f"image_{image_number:03d}.png"
            )
            output_path = assets_dir / output_name

            image.save(output_path)
            saved_paths.append(output_path.as_posix())

            markdown_text = markdown_text.replace(
                str(source_name),
                output_path.as_posix(),
            )

        if append_missing:
            missing_images = [
                path for path in saved_paths if path not in markdown_text
            ]
            if missing_images:
                image_markdown = "\n\n".join(
                    f"![Изображение страницы {page_index + 1}]({path})"
                    for path in missing_images
                )
                markdown_text = PaddleDocumentAnalyzer._append_section(
                    markdown_text,
                    "Изображения страницы",
                    image_markdown,
                )

        return markdown_text

    @staticmethod
    def _save_native_page_images(
        markdown_text: str,
        page,
        assets_dir: Path,
        page_index: int,
    ) -> str:
        """Сохраняет только исходные растровые объекты цифрового PDF."""
        image_links = []
        seen_xrefs = set()
        for image_number, image_info in enumerate(
            page.get_images(full=True),
            start=1,
        ):
            xref = image_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            extracted = page.parent.extract_image(xref)
            extension = extracted.get("ext", "png")
            output_path = assets_dir / (
                f"page_{page_index + 1:04d}_"
                f"image_{image_number:03d}.{extension}"
            )
            output_path.write_bytes(extracted["image"])
            image_links.append(
                f"![Изображение страницы {page_index + 1}]"
                f"({output_path.as_posix()})"
            )

        if image_links:
            return PaddleDocumentAnalyzer._append_section(
                markdown_text,
                "Изображения страницы",
                "\n\n".join(image_links),
            )
        return markdown_text

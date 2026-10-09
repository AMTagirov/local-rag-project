import re
import unicodedata
from typing import Generator, Optional

from src.config.schema import DocumentAnalysisConfig

import pymupdf as fitz
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

from src.core.interfaces import DocumentParser


def normalize_document_text(text: str) -> str:
    """Мягкая очистка текста без потери регистра, пунктуации и абзацев."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")  # мягкий перенос

    # Склеиваем слова, разорванные переносом строки: "инфор-\nмация" -> "информация".
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", text)

    # Удаляем служебные и невидимые Unicode-символы, сохраняя переносы строк.
    text = "".join(
        char
        for char in text
        if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf"}
    )

    normalized_paragraphs = []
    for paragraph in re.split(r"\n[ \t]*\n+", text):
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in paragraph.split("\n")]
        normalized = " ".join(line for line in lines if line)
        if normalized:
            normalized_paragraphs.append(normalized)

    return "\n\n".join(normalized_paragraphs)

class PDFDocumentParser(DocumentParser):
    """
    Извлекает PDF обычным PyMuPDF или через PP-StructureV3.
    """

    def __init__(
        self,
        analysis_config: Optional[
            DocumentAnalysisConfig
        ] = None,
    ):
        self.analysis_config = analysis_config
        self._analyzer = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            from src.services.document_analysis_service import (
                PaddleDocumentAnalyzer,
            )

            self._analyzer = PaddleDocumentAnalyzer(
                self.analysis_config
            )

        return self._analyzer

    def parse(
        self,
        file_path: str,
    ) -> Generator[str, None, None]:
        if (
            self.analysis_config
            and self.analysis_config.enabled
        ):
            yield from self.analyzer.parse_pdf(file_path)
            return

        yield from self._parse_with_pymupdf(file_path)

    def _parse_with_pymupdf(
        self,
        file_path: str,
    ) -> Generator[str, None, None]:
        has_text = False

        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    paragraphs = []

                    for block in page.get_text(
                        "blocks",
                        sort=True,
                    ):
                        if (
                            len(block) > 6
                            and block[6] != 0
                        ):
                            continue

                        cleaned_block = (
                            normalize_document_text(
                                str(block[4])
                            )
                        )

                        if cleaned_block:
                            paragraphs.append(
                                cleaned_block
                            )

                    if paragraphs:
                        has_text = True
                        yield "\n\n".join(paragraphs)

            if not has_text:
                raise RuntimeError(
                    f"Не удалось извлечь текст из "
                    f"'{file_path}'. Возможно, документ "
                    f"не содержит текстового слоя."
                )

        except Exception as error:
            if isinstance(error, RuntimeError):
                raise

            raise RuntimeError(
                f"Ошибка при чтении PDF-файла "
                f"'{file_path}': {error}"
            ) from error

class DocxDocumentParser(DocumentParser):
    """Потоковый парсер для документов Microsoft Word (.docx)."""

    def parse(self, file_path: str) -> Generator[str, None, None]:
        """
        Читает абзацы и таблицы в порядке документа и отдаёт их блоками,
        чтобы не загружать весь текст книги одновременно в память.
        """
        try:
            doc = Document(file_path)
            buffer = []
            elements_per_block = 5
            has_content = False

            for element in self._iter_document_elements(doc):
                if isinstance(element, Paragraph):
                    text = normalize_document_text(element.text)
                else:
                    text = self._table_to_markdown(element)

                if not text:
                    continue

                has_content = True
                buffer.append(text)

                if len(buffer) >= elements_per_block:
                    yield "\n\n".join(buffer)
                    buffer = []

            if buffer:
                yield "\n\n".join(buffer)

            if not has_content:
                raise RuntimeError(
                    f"Word-файл '{file_path}' не содержит текста или таблиц."
                )

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Ошибка при чтении Word файла '{file_path}': {e}")

    @staticmethod
    def _iter_document_elements(doc):
        """Возвращает абзацы и таблицы в исходном порядке DOCX."""
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, doc)
            elif isinstance(child, CT_Tbl):
                yield Table(child, doc)

    @staticmethod
    def _table_to_markdown(table: Table) -> str:
        rows = []
        for row in table.rows:
            cells = [
                normalize_document_text(cell.text)
                .replace("\n", "<br>")
                .replace("|", "\\|")
                for cell in row.cells
            ]
            if any(cells):
                rows.append(cells)

        if not rows:
            return ""

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        header = rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        lines.extend(
            "| " + " | ".join(row) + " |" for row in rows[1:]
        )
        return "\n".join(lines)



from src.config.schema import RAGConfig


def create_document_parser(
    config: RAGConfig,
) -> DocumentParser:
    """
    Создаёт основной PDF-парсер с настройками анализа.
    DOCX в режиме auto выбирается внутри RAGPipeline.
    """
    return PDFDocumentParser(
        analysis_config=config.document_analysis
    )

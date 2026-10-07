import re
import unicodedata
from typing import Generator

import pymupdf as fitz
from docx import Document

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
    """Промышленная потоковая реализация парсинга PDF без перегрузки RAM."""

    def parse(self, file_path: str) -> Generator[str, None, None]:
        """
        Потоково извлекает текст из PDF файла (постранично).
        """
        has_text = False
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    paragraphs = []
                    for block in page.get_text("blocks", sort=True):
                        # В PyMuPDF поле block_type == 0 обозначает текст, 1 — изображение.
                        if len(block) > 6 and block[6] != 0:
                            continue
                        cleaned_block = normalize_document_text(str(block[4]))
                        if cleaned_block:
                            paragraphs.append(cleaned_block)

                    if paragraphs:
                        has_text = True
                        yield "\n\n".join(paragraphs)
            
            if not has_text:
                raise RuntimeError(
                    f"Не удалось извлечь текст из '{file_path}'. "
                    "Возможно, это отсканированный документ без текстового слоя (OCR)."
                )
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Ошибка при чтении PDF файла '{file_path}': {e}")

class DocxDocumentParser:
    """Потоковый парсер для документов Microsoft Word (.docx)."""

    def parse(self, file_path: str) -> Generator[str, None, None]:
        """
        Лениво читает Word-файл по абзацам и отдает их блоками,
        чтобы не загружать весь текст книги одновременно в память.
        """
        try:
            doc = Document(file_path)
            buffer = []
            paragraphs_per_block = 5  # Группируем по 5 абзацев в один "экран" текста

            for paragraph in doc.paragraphs:
                text = normalize_document_text(paragraph.text)
                if text:
                    buffer.append(text)
                
                # Как только накопили блок абзацев — отдаем его в сплиттер
                if len(buffer) >= paragraphs_per_block:
                    # Объединяем через двойной перенос, чтобы семантический сплиттер видел границы
                    yield "\n\n".join(buffer)
                    buffer = []

            # Отдаем оставшийся хвост документа
            if buffer:
                yield "\n\n".join(buffer)

        except Exception as e:
            raise RuntimeError(f"Ошибка при чтении Word файла '{file_path}': {e}")

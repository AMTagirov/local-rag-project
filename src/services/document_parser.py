from typing import Generator
import fitz  # PyMuPDF
from src.core.interfaces import DocumentParser

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
                    page_text = page.get_text().strip()
                    if page_text:
                        has_text = True
                        
                        # 1. Заменяем реальные абзацы временным уникальным маркером
                        cleaned = page_text.replace("\n\n", "||PARAGRAPH||")
                        # 2. Убираем одиночные переносы строк (которые ломают предложения посреди строки)
                        cleaned = cleaned.replace("\n", " ")
                        # 3. Возвращаем правильные двойные переносы строк на место
                        cleaned = cleaned.replace("||PARAGRAPH||", "\n\n")
                        # 4. Схлопываем случайные множественные пробелы
                        cleaned = " ".join(cleaned.split())
                        # 5. Возвращаем правильный формат для абзацев
                        cleaned = cleaned.replace(". ", ".\n\n") # Опционально: гарантирует, что точки станут границами
                        
                        yield cleaned
            
            if not has_text:
                raise RuntimeError(
                    f"Не удалось извлечь текст из '{file_path}'. "
                    "Возможно, это отсканированный документ без текстового слоя (OCR)."
                )
            if not has_text:
                raise RuntimeError(f"Не удалось извлечь текст из '{file_path}'.")
        except Exception as e:
            if isinstance(e, RuntimeError): raise e
            raise RuntimeError(f"Ошибка при чтении PDF файла '{file_path}': {e}")
        
        

from docx import Document

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
                text = paragraph.text.strip()
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
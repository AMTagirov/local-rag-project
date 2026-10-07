import gc
import numpy as np
import re
from typing import List, Generator, Iterable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.interfaces import TextSplitter
from src.config.schema import ChunkingStrategy, SplitterConfig
from src.core.interfaces import EmbeddingModel

class ChunkTextSplitter(TextSplitter):
    def __init__(self, config: SplitterConfig, embedder: EmbeddingModel):
        self.config = config
        self.embedder = embedder
        
        # Страховочный сплиттер для слишком длинных смысловых блоков
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.buffer_size = 3
        # Оптимизировано: если block_size не задан, для учебников лучше использовать 25-30
        self.block_size = config.block_size if hasattr(config, "block_size") else 25
        self.threshold_percentile = config.breakpoint_percentile_threshold if hasattr(config, "breakpoint_percentile_threshold") else 85
        
        # Переменная для хранения текущего заголовка/контекста верхнего уровня
        self.current_header = ""

    def _is_header(self, text: str) -> bool:
        """
        Эвристическая проверка: является ли абзац заголовком.
        Подходит для учебников (Главы, Параграфы, Нумерация, Капс).
        """
        clean_text = text.strip()
        if not clean_text:
            return False

        # Длинная строка без пробелов или в верхнем регистре — это текст,
        # а не заголовок, даже если в ней мало "слов".
        if len(clean_text) > 200:
            return False
            
        # Правило 1: Короткая строка (меньше 12 слов) и начинается с цифр (1., 1.2, Глава 3, Параграф)
        header_patterns = [
            r'^(Глава|§|Параграф|Раздел|Часть)\s+\d+', 
            r'^\d+(\.\d+)*\s+[А-ЯA-Z]', 
        ]
        word_count = len(clean_text.split())
        
        if word_count < 12:
            for pattern in header_patterns:
                if re.search(pattern, clean_text, re.IGNORECASE):
                    return True
                    
        # Правило 2: Текст написан преимущественно в верхнем регистре (КАПСОМ) и он не слишком длинный
        if clean_text.isupper() and word_count < 10:
            return True
            
        return False

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return list(self.split_stream([text]))

    def split_stream(self, pages_iterator: Iterable[str]) -> Generator[str, None, None]:
        """
        Потоково разбивает страницы согласно выбранной стратегии.
        """
        if self.config.strategy == ChunkingStrategy.RECURSIVE:
            yield from self._split_recursive_stream(pages_iterator)
            return

        current_paragraphs_block = []
        self.current_header = "Общий контекст"  # Базовый заголовок по умолчанию

        for page_text in pages_iterator:
            if not page_text:
                continue
            
            # Нарезаем страницу на абзацы
            page_paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            current_paragraphs_block.extend(page_paragraphs)

            # Обрабатываем накопившиеся блоки
            while len(current_paragraphs_block) >= self.block_size:
                block_to_process = current_paragraphs_block[:self.block_size]
                current_paragraphs_block = current_paragraphs_block[self.block_size:]

                yield from self._process_local_block(block_to_process)
                
                del block_to_process
                gc.collect()

        # Обрабатываем оставшиеся "хвосты" после завершения книги
        if current_paragraphs_block:
            yield from self._process_local_block(current_paragraphs_block)

    def _split_recursive_stream(self, pages_iterator: Iterable[str]) -> Generator[str, None, None]:
        """Рекурсивно режет поток, сохраняя overlap между страницами."""
        pending_text = ""

        for page_text in pages_iterator:
            if not page_text or not page_text.strip():
                continue

            pending_text = "\n\n".join(
                part for part in (pending_text, page_text.strip()) if part
            )
            chunks = self.fallback_splitter.split_text(pending_text)

            # Последний чанк может ещё вырасти за счёт следующей страницы.
            # Остальные чанки уже завершены и их можно отдать сразу.
            if len(chunks) > 1:
                yield from chunks[:-1]
                pending_text = chunks[-1]

        if pending_text:
            yield from self.fallback_splitter.split_text(pending_text)

    def _process_local_block(self, paragraphs: List[str]) -> Generator[str, None, None]:
        """Локальный семантический анализ пачки абзацев."""
        if len(paragraphs) == 0:
            return
            
        if len(paragraphs) == 1:
            # Обновляем заголовок, если единственный абзац им является
            paragraph = paragraphs[0]
            if self._is_header(paragraph):
                self.current_header = paragraph

            # Один длинный абзац тоже должен соблюдать chunk_size.
            chunks = (
                self.fallback_splitter.split_text(paragraph)
                if len(paragraph) > self.config.chunk_size
                else [paragraph]
            )
            for chunk in chunks:
                yield f"[{self.current_header}]\n{chunk}"
            return

        # Перед эмбеддингами сканируем блок и обновляем текущий заголовок, 
        # чтобы скользящие окна buffer_size уже учитывали актуальную тему
        for p in paragraphs[:3]: 
            if self._is_header(p):
                self.current_header = p

        windows = [" ".join(paragraphs[max(0, i - self.buffer_size) : i + 1]) for i in range(len(paragraphs))]
        embeddings = np.array(self.embedder.embed_documents(windows))
        del windows

        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        normalized = embeddings / norm
        distances = 1.0 - np.sum(normalized[:-1] * normalized[1:], axis=1)
        del embeddings

        threshold = np.percentile(distances, self.threshold_percentile)
        split_indices = set(np.where(distances > threshold)[0].tolist())

        current_chunk = []
        
        for i, paragraph in enumerate(paragraphs):
            # Если внутри блока встретили новый заголовок, актуализируем контекст
            if self._is_header(paragraph):
                # Если в буфере уже что-то накопилось, принудительно отдаем старый чанк перед сменой темы
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk).strip()
                    yield f"[{self.current_header}]\n{chunk_text}"
                    current_chunk = []
                self.current_header = paragraph

            current_chunk.append(paragraph)
            chunk_text = "\n\n".join(current_chunk).strip()
            
            # Проверяем условия реза (семантический разрыв ИЛИ превышение лимита символов)
            if i in split_indices or len(chunk_text) >= self.config.chunk_size:
                if chunk_text:
                    # Если блок получился аномально большим, пускаем страховочный сплиттер
                    if len(chunk_text) > self.config.chunk_size * 1.8:
                        sub_chunks = self.fallback_splitter.split_text(chunk_text)
                        for sub_c in sub_chunks:
                            yield f"[{self.current_header}]\n{sub_c}"
                    else:
                        yield f"[{self.current_header}]\n{chunk_text}"
                current_chunk = []

        # Сливаем остаток
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk).strip()
            if chunk_text: 
                yield f"[{self.current_header}]\n{chunk_text}"

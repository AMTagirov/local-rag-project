from typing import List, Any
from fastembed import SparseTextEmbedding

class SparseEmbeddingService:
    """
    Сервис для генерации разреженных (sparse) векторов.
    Используется для реализации поиска по ключевым словам (BM25-like).
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        """
        Инициализирует модель.
        """
        # Передаем model_name в конструктор, чтобы избежать ошибки
        self.model = SparseTextEmbedding(model_name=model_name, threads=4)

    def embed(self, texts: List[str]) -> List[Any]:
        """
        Преобразует список текстов в список объектов разреженных векторов документов.
        Используется строго при индексации (инжесте) файлов.
        """
        # model.embed возвращает генератор объектов fastembed
        return list(self.model.embed(texts))

    def embed_query(self, query: str) -> List[Any]:
        """
        Преобразует строку поискового запроса в разреженный вектор запроса.
        Используется строго внутри поисковых пайплайнов (query_with_contexts).
        """
        # ВАЖНО: query_embed выставляет фиксированные веса токенов для поиска.
        # Метод ожидает список строк, поэтому оборачиваем query в список [query].
        return list(self.model.query_embed([query]))
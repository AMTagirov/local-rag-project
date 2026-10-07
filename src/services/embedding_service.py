from sentence_transformers import SentenceTransformer
from src.core.interfaces import EmbeddingModel
from src.config.schema import EmbeddingConfig
from typing import List

class EmbeddingService(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        # Загрузка локальной модели (например, intfloat/multilingual-e5-large)
        self.model = SentenceTransformer(config.model_name)

    def embed_query(self, text: str) -> List[float]:
        """
        Преобразует строку поискового запроса пользователя в плотный вектор.
        Применяет обязательный префикс 'query: '.
        """
        if not text or not text.strip():
            return []
            
        processed_text = f"query: {text}"
        embedding = self.model.encode(
            processed_text,
            normalize_embeddings=True, # Гарантирует корректность Cosine поиска в Qdrant
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Преобразует пачку чанков документов в матрицу плотных векторов для инжеста в БД.
        Применяет обязательный префикс 'passage: '.
        """
        if not texts:
            return []
        
        prefixed_texts = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(
            prefixed_texts, 
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed(self, text: str, is_query: bool = True) -> List[float]:
        """
        Устаревший метод для обратной совместимости с базовыми интерфейсами.
        По умолчанию считает, что одиночный вызов — это поисковый запрос.
        """
        if is_query:
            return self.embed_query(text)
        return self.embed_documents([text])[0]

    def get_dimension(self) -> int:
        return self.config.dimension
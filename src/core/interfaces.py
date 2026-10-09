from abc import ABC, abstractmethod
from typing import List, Any, Iterator, Optional

class DocumentParser(ABC):
    """Интерфейс для парсинга документов."""
    @abstractmethod
    def parse(self, file_path: str) -> Iterator[str]:
        pass

class TextSplitter(ABC):
    """Интерфейс для разбиения текста на чанки."""
    @abstractmethod
    def split(self, text: str) -> List[str]:
        pass

class EmbeddingModel(ABC):
    """Интерфейс для модели эмбеддингов."""
    @abstractmethod
    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """Добавили is_query для поддержки моделей типа E5."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Добавили пакетный эмбеддинг."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass

class VectorStore(ABC):
    """Интерфейс для векторного хранилища."""
    @abstractmethod
    def create_collection(self, name: Optional[str] = None, dimension: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def collection_exists(self, name: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    def upsert(self, points: List[Any], collection_name: Optional[str] = None) -> None:
        """Исправлено: points идет первым, так как это обязательный аргумент."""
        pass

    @abstractmethod
    def search(
        self, 
        collection_name: Optional[str] = None, 
        query_text: Optional[str] = None, 
        query_vector: Optional[List[float]] = None, 
        limit: int = 10
    ) -> List[Any]:
        pass

class LLMService(ABC):
    """Интерфейс для языковой модели (LLM)."""
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass

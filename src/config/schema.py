import os
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import yaml

class EmbeddingConfig(BaseModel):
    model_name: str = Field(..., description="Название модели эмбеддингов (например, intfloat/multilingual-e5-large)")
    dimension: int = Field(..., description="Размерность вектора")

class LLMConfig(BaseModel):
    model_name: str = Field(..., description="Название модели в Ollama (например, qwen2.5:14b)")
    base_url: str = Field("http://localhost:11434", description="URL адрес Ollama")
    num_ctx: int = Field(8192, ge=2048, description="Размер контекстного окна для Ollama")  # Добавили важный параметр!
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Температура генерации. 0.0 — жесткий детерминизм.")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Параметр top_p фильтрации токенов.")
    seed: int = Field(42, description="Сид для генератора случайных чисел модели.")
    
class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"  # Классический текстовый сплиттер
    SEMANTIC = "semantic"    # Умный сплиттер по смыслу

class SplitterConfig(BaseModel):
    chunk_size: int = Field(1500, ge=100, description="Максимальный размер чанка")
    chunk_overlap: int = Field(300, ge=0, description="Размер перекрытия")
    
    # НОВЫЕ ПОЛЯ ДЛЯ СЕМАНТИЧЕСКОГО СЧЕТА:
    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SEMANTIC, 
        description="Стратегия чанкинга: recursive или semantic"
    )
    breakpoint_percentile_threshold: int = Field(
        default=92, 
        ge=50, le=100, 
        description="Перцентиль порога разрыва темы. Чем выше, тем крупнее чанки."
    )
    block_size: int = Field(
        default=100, 
        ge=10, 
        description="Размер блока абзацев для защиты от OOM на больших файлах."
    )

    @field_validator("chunk_overlap")
    @classmethod
    def check_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size and v >= chunk_size:
            raise ValueError("chunk_overlap должен быть строго меньше chunk_size")
        return v


from enum import Enum
from pydantic import BaseModel

class SearchMode(str, Enum):
    DENSE = "dense"       # Только семантика
    SPARSE = "sparse"     # Только ключевые слова
    HYBRID = "hybrid"     # Семантика + Ключевые слова (RRF)

class VectorStoreConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str
    url: Optional[str] = None
    docs_dir: str = "data/docs"  # Путь к папке с документами
    recreate_on_start: bool = False  # Флаг пересоздания коллекции
    search_mode: SearchMode = SearchMode.HYBRID  # По умолчанию гибрид

class RerankerConfig(BaseModel):
    model_name: str = Field("BAAI/bge-reranker-base", description="Модель Cross-Encoder для переранжирования")
    use_reranker: bool = Field(True, description="Использовать ли этап переранжирования")
    top_n_retrieval: int = Field(20, ge=1, description="Сколько документов достать из БД для последующего переранжирования")


class RAGConfig(BaseModel):
    """Главный объект конфигурации всего пайплайна (Pydantic V2)."""
    experiment_name: str = "default_experiment"
    parser_type: str = "pdf"  # 'pdf' или 'text'
    top_k: int = Field(3, ge=1)
    
    embedding: EmbeddingConfig
    llm: LLMConfig
    splitter: SplitterConfig
    vector_store: VectorStoreConfig
    reranker: RerankerConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RAGConfig":
        """Загрузка конфигурации из YAML файла с валидацией Pydantic V2."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Конфиг не найден по пути: {yaml_path}")
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        # В Pydantic V2 вместо cls(**config_dict) используется метод model_validate
        return cls.model_validate(config_dict)
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


class ParserType(str, Enum):
    AUTO = "auto"
    PDF = "pdf"
    DOCX = "docx"

class SplitterConfig(BaseModel):
    chunk_size: int = Field(1500, ge=100, description="Максимальный размер чанка")
    chunk_min_size: int = Field(400, ge=1, description="Минимальный желаемый размер чанка")
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

    @field_validator("chunk_min_size")
    @classmethod
    def check_min_chunk_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size and v * 2 > chunk_size:
            raise ValueError("chunk_min_size должен быть не больше половины chunk_size")
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
    model_name: str = Field("BAAI/bge-reranker-v2-m3", description="Модель Cross-Encoder для переранжирования")
    use_reranker: bool = Field(True, description="Использовать ли этап переранжирования")
    top_n_retrieval: int = Field(20, ge=1, description="Сколько документов достать из БД для последующего переранжирования")


class QueryRewritingConfig(BaseModel):
    enabled: bool = Field(False, description="Переписывать запрос перед поиском")
    keep_original: bool = Field(
        True,
        description="Искать также по исходному запросу и объединять результаты через RRF",
    )
    rrf_k: int = Field(60, ge=1, description="Константа RRF для объединения выдач")


class DocumentAnalysisConfig(BaseModel):
    enabled: bool = Field(
        False,
        description="Использовать PP-StructureV3 для анализа PDF",
    )

    device: str = Field(
        "cpu",
        description="Устройство PaddleOCR: cpu или gpu",
    )
    engine: str = Field(
        "onnxruntime",
        description="Движок OCR; ONNX Runtime обходит CPU-ошибку Paddle PIR/oneDNN",
    )
    ocr_language: str = Field(
        "ru",
        description="Язык OCR PaddleOCR",
    )
    ocr_version: str = Field(
        "PP-OCRv5",
        description="Версия OCR-моделей PaddleOCR",
    )
    prefer_native_pdf_text: bool = Field(
        True,
        description="Сохранять встроенный текст цифрового PDF вместо повторного OCR",
    )
    native_text_min_chars: int = Field(
        80,
        ge=1,
        description="Минимум символов для признания страницы цифровой",
    )

    use_formula_recognition: bool = False
    formula_model_name: str = Field(
        "PP-FormulaNet_plus-M",
        description="Модель формул; plus-M экономнее памяти, чем plus-L",
    )
    formula_filter_enabled: bool = Field(
        True,
        description="Распознавать формулы только на отобранных страницах",
    )
    formula_filter_threshold: int = Field(
        2,
        ge=1,
        le=4,
        description="Минимальный комбинированный балл страницы с формулами",
    )
    formula_render_dpi: int = Field(
        250,
        ge=96,
        le=300,
        description="Разрешение повторного рендера страницы с формулами",
    )
    use_table_recognition: bool = False
    use_chart_recognition: bool = False

    save_images: bool = True

    assets_dir: str = Field(
        "data/extracted",
        description="Каталог для изображений, извлечённых из документов",
    )
    cache_dir: str = Field(
        "data/models/paddlex",
        description="Локальный каталог моделей и кэша PaddleX",
    )

class RAGConfig(BaseModel):
    """Главный объект конфигурации всего пайплайна (Pydantic V2)."""
    experiment_name: str = "default_experiment"
    parser_type: ParserType = ParserType.AUTO
    top_k: int = Field(3, ge=1)
    
    embedding: EmbeddingConfig
    llm: LLMConfig
    splitter: SplitterConfig
    vector_store: VectorStoreConfig
    reranker: RerankerConfig
    query_rewriting: QueryRewritingConfig = Field(default_factory=QueryRewritingConfig)
    document_analysis: DocumentAnalysisConfig = Field(
        default_factory=DocumentAnalysisConfig
    )


    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RAGConfig":
        """Загрузка конфигурации из YAML файла с валидацией Pydantic V2."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Конфиг не найден по пути: {yaml_path}")
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        # В Pydantic V2 вместо cls(**config_dict) используется метод model_validate
        return cls.model_validate(config_dict)

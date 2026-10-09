from typing import List, Optional, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams 
from src.core.interfaces import VectorStore
from src.config.schema import VectorStoreConfig, SearchMode
from src.services.sparse_embedding_service import SparseEmbeddingService
from src.core.interfaces import EmbeddingModel


class QdrantService(VectorStore):
    """
    Сервис для работы с векторным хранилищем Qdrant.
    Реализует интерфейс VectorStore на основе VectorStoreConfig.
    """

    def __init__(self, config: VectorStoreConfig, embedder: EmbeddingModel):
        self.config = config
        self.embedder = embedder  
        self.sparse_service = SparseEmbeddingService()
        
        try:
            if config.url:
                self.client = QdrantClient(url=config.url)
            else:
                self.client = QdrantClient(host=config.host, port=config.port)
        except Exception as e:
            print(f"⚠️ Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self, name: Optional[str] = None, dimension: Optional[int] = None) -> None:
        """
        Создает новую коллекцию с поддержкой плотных и разреженных векторов (BM25).
        """
        collection_name = name or self.config.collection_name
        
        if dimension is None:
            raise ValueError("Dimension must be provided to create a collection.")

        try:
            if not self.collection_exists(collection_name):
                print(f"🏗️ Создание коллекции '{collection_name}' с поддержкой BM25 (Modifier.IDF)...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "default": VectorParams(
                            size=dimension,
                            distance=Distance.COSINE
                        )
                    },
                    # ИСПРАВЛЕНО: передаем "idf" строкой — это работает во всех версиях qdrant-client
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            modifier="idf"
                        )
                    }
                )
                print(f"✅ Коллекция '{collection_name}' успешно создана.")
        except Exception as e:
            print(f"⚠️ Error creating collection '{collection_name}': {e}")
            raise

    def delete_collection(self, name: Optional[str] = None) -> None:
        collection_name = name or self.config.collection_name
        try:
            if self.collection_exists(collection_name):
                self.client.delete_collection(collection_name=collection_name)
                print(f"🗑️ Коллекция '{collection_name}' удалена.")
            else:
                print(f"ℹ️ Коллекция '{collection_name}' не существует, удаление пропущено.")
        except Exception as e:
            print(f"⚠️ Не удалось удалить коллекцию '{collection_name}': {e}")
            raise

    def collection_exists(self, name: Optional[str] = None) -> bool:
        collection_name = name or self.config.collection_name
        try:
            return self.client.collection_exists(collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Не удалось проверить коллекцию '{collection_name}' в Qdrant: {e}"
            ) from e

    def upsert(self, collection_name: Optional[str] = None, points: List[PointStruct] = None) -> None:
        target_collection = collection_name or self.config.collection_name
        if points is None:
            raise ValueError("Points list must be provided for upsert.")
            
        try:
            self.client.upsert(collection_name=target_collection, points=points)
        except Exception as e:
            print(f"⚠️ Error upserting points to collection '{target_collection}': {e}")
            raise
    
    def search(
        self, 
        collection_name: Optional[str] = None, 
        query_text: Optional[str] = None, 
        query_vector: Optional[List[float]] = None, 
        limit: int = 10
    ) -> List[Any]:
        target_collection = collection_name or self.config.collection_name
        mode = self.config.search_mode
        
        dense_vec = query_vector
        sparse_vec = None

        if query_text is not None:
            # 1. ПОДГОТОВКА ПЛОТНОГО ВЕКТОРА (Исправлено)
            if mode in [SearchMode.DENSE, SearchMode.HYBRID]:
                # Явно вызываем embed_query, гарантируя префикс "query: " при поиске
                if hasattr(self.embedder, 'embed_query'):
                    dense_vec = self.embedder.embed_query(query_text)
                else:
                    dense_vec = self.embedder.embed(query_text, is_query=True)
            
            # 2. ПОДГОТОВКА РАЗРЕЖЕННОГО ВЕКТОРА (BM25)
            if mode in [SearchMode.SPARSE, SearchMode.HYBRID]:
                sparse_output = self.sparse_service.embed_query(query_text)[0]
                
                if hasattr(sparse_output, "indices") and hasattr(sparse_output, "values"):
                    sparse_vec = models.SparseVector(
                        indices=list(sparse_output.indices),
                        values=list(sparse_output.values)
                    )
                elif hasattr(sparse_output, "as_object"):
                    sparse_vec = sparse_output.as_object()

        if dense_vec is None and mode != SearchMode.SPARSE:
            raise ValueError("Either query_text or query_vector must be provided.")

        if mode == SearchMode.DENSE:
            return self._search_dense(target_collection, dense_vec, limit)
        elif mode == SearchMode.SPARSE:
            return self._search_sparse(target_collection, sparse_vec, limit)
        elif mode == SearchMode.HYBRID:
            return self._search_hybrid(target_collection, dense_vec, sparse_vec, limit)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
    
    def _search_dense(self, collection_name: str, vector: List[float], limit: int):
        # Для обычного одиночного поиска по умолчанию передаем чистый список float
        # Если коллекция строго требует именованный вектор, используем models.NamedVector
        response = self.client.query_points(
            collection_name=collection_name,
            query=vector,  # Передаем вектор напрямую, без обертки в словарь
            using="default",  # Указываем пространство через параметр using
            limit=limit
        )
        return response.points
        
    def _search_sparse(self, collection_name: str, sparse_vector: models.SparseVector, limit: int):
        # Для поиска по ключевым словам передаем чистый объект SparseVector
        response = self.client.query_points(
            collection_name=collection_name,
            query=sparse_vector,  # Передаем разреженный вектор напрямую
            using="sparse",  # Указываем пространство ключевых слов
            limit=limit
        )
        return response.points
            
    def _search_hybrid(self, collection_name: str, dense_vector: List[float], sparse_vector: models.SparseVector, limit: int):
        # ИСПРАВЛЕНО ДЛЯ GATHER/Pydantic:
        # Внутри Prefetch для именованных конфигураций векторы и пространства передаются 
        # раздельно через параметры query и using. 
        # Корневой query принимает только FusionQuery, никаких сырых векторов на верхнем уровне!
        response = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(query=dense_vector, using="default", limit=limit),
                models.Prefetch(query=sparse_vector, using="sparse", limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        return response.points
    
    def get_existing_file_names(self, collection_name: Optional[str] = None) -> set:
        target_collection = collection_name or self.config.collection_name
        try:
            if not self.collection_exists(target_collection):
                return set()

            file_names = set()
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=target_collection,
                    limit=256,
                    offset=offset,
                    with_payload=["file_name"],
                    with_vectors=False,
                )
                file_names.update(
                    point.payload.get("file_name")
                    for point in points
                    if point.payload and point.payload.get("file_name")
                )
                if offset is None:
                    break
            return file_names
        except Exception as e:
            print(f"⚠️ Ошибка при проверке существующих файлов: {e}")
            return set()

    def delete_by_file_name(
        self,
        file_name: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """Удаляет все чанки документа; используется для отката неудачной загрузки."""
        target_collection = collection_name or self.config.collection_name
        if not self.collection_exists(target_collection):
            return
        self.client.delete(
            collection_name=target_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_name",
                            match=models.MatchValue(value=file_name),
                        )
                    ]
                )
            ),
            wait=True,
        )
    
    def _get_already_indexed_files(self, collection_name: str) -> set:
            """
            Сканирует базу данных, чтобы найти уникальные имена файлов в payload.
            """
            try:
                # Используем scroll (пролистывание) для получения всех point'ов
                
                # но для локального RAG scroll работает отлично.
                points, _ = self.vector_store.client.scroll(
                    collection_name=collection_name,
                    limit=10000 # Лимит на проверку
                )
                return {p.payload.get("file_name") for p in points if p.payload.get("file_name")}
            except Exception as e:
                print(f"⚠️ Ошибка при проверке существующих файлов: {e}")
                return set()        














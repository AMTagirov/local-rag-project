import time
import os
import glob
from typing import List, Any, Dict, Optional, Tuple 
import uuid 
import gc
from itertools import islice

from src.core.interfaces import (
    DocumentParser, 
    TextSplitter, 
    EmbeddingModel, 
    VectorStore, 
    LLMService
)
from src.config.schema import RAGConfig, SearchMode
from src.services.reranker_service import RerankerService 
from qdrant_client.models import PointStruct, SparseVector 

class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        parser: DocumentParser,
        splitter: TextSplitter,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        llm: LLMService,
        reranker: Optional[RerankerService] = None,
        sparse_service: Any = None
    ):
        self.config = config
        self.parser = parser
        self.splitter = splitter
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker 
        self.sparse_service = sparse_service
        
    def sync_directory(self, force_recreate: bool = False):
        """
        Сканирует папку, загружает новые документы и пропускает уже существующие.
        """
        docs_dir = self.config.vector_store.docs_dir
        collection_name = self.config.vector_store.collection_name
        
        if force_recreate or self.config.vector_store.recreate_on_start:
            print(f"🔄 Режим полной синхронизации. Пересоздание коллекции '{collection_name}'...")
            self.vector_store.delete_collection(collection_name)
            force_recreate = True

        if not os.path.exists(docs_dir):
            print(f"⚠️ Папка '{docs_dir}' не найдена. Создаю её...")
            os.makedirs(docs_dir)
            return

        # 2. Получаем список всех поддерживаемых файлов
        extensions = ['*.pdf', '*.docx']
        files_to_process = []
        for ext in extensions:
            files_to_process.extend(glob.glob(os.path.join(docs_dir, ext)))

        if not files_to_process:
            print(f"ℹ️ В папке '{docs_dir}' нет документов.")
            return

        print(f"🔍 Найдено файлов в папке: {len(files_to_process)}")

        # 3. Инициализируем коллекцию, если её нет
        dimension = self.embedder.get_dimension()
        self.vector_store.create_collection(collection_name, dimension)

        # ИСПОЛЬЗУЕМ АБСТРАКЦИЮ: получаем список файлов через интерфейс VectorStore
        existing_files = self.vector_store.get_existing_file_names(collection_name)
        print(f"📊 Уже в базе: {len(existing_files)}, Нужно загрузить: {len(files_to_process) - len(existing_files)}")

        for file_path in files_to_process:
            file_name = os.path.basename(file_path)
            if file_name in existing_files:
                print(f"⏭️ Пропуск: '{file_name}' уже проиндексирован.")
                continue

            print(f"📥 Загрузка нового документа: '{file_name}'...")
            self._ingest_single_file(file_path, collection_name)



    def _ingest_single_file(self, file_path: str, collection_name: str, batch_size: int = 512):
        """
        Финальный продакшен-метод загрузки файлов любого объема (включая 2000+ страниц).
        Работает в полностью потоковом режиме. RAM стабильна.
        """
        file_name = os.path.basename(file_path)
        print(f"📖 [Pipeline] Запущен сквозной стриминг для файла: {file_name}")
        
        # 1. Получаем ленивый генератор страниц из парсера
        if file_path.endswith('.docx'):
            from src.services.document_parser import DocxDocumentParser
            active_parser = DocxDocumentParser()
        else:
            active_parser = self.parser  # По умолчанию PDFDocumentParser
        
        pages_stream = active_parser.parse(file_path) 
        # 2. Передаем генератор страниц в семантический сплиттер
        # Получаем генератор готовых смысловых чанков текста
        chunks_stream = self.splitter.split_stream(pages_stream)
        
        start_idx = 0
        while True:
            # 3. Нарезаем поток чанков на батчи по 512 элементов для отправки в БД
            batch_chunks = list(islice(chunks_stream, batch_size))
            if not batch_chunks:
                break  # Поток полностью вычитан, книга закончилась

            end_idx = start_idx + len(batch_chunks)
            print(f"🧠 [Embedding] Обработка смысловых чанков {start_idx} -> {end_idx}...")
            
            try:
                # Генерация плотных векторов (батч контролируется параметром функции)
                batch_dense_vectors = self.embedder.embed_documents(batch_chunks)
                
                # Генерация разреженных векторов (BM25 считает эту пачку за миллисекунды)
                batch_sparse_vectors = []
                if self.sparse_service:
                    batch_sparse_vectors = self.sparse_service.embed(batch_chunks)
                    
                print(f"📦 [VectorStore] Отправка {len(batch_chunks)} точек в Qdrant...")
                points = []
                
                for i, chunk in enumerate(batch_chunks):
                    dense_vec = batch_dense_vectors[i]
                    if hasattr(dense_vec, "tolist"):
                        dense_vec = dense_vec.tolist()
                    elif not isinstance(dense_vec, list):
                        dense_vec = list(dense_vec)
                        
                    vector_dict = {"default": dense_vec}
                    
                    if self.sparse_service and batch_sparse_vectors:
                        s_obj = batch_sparse_vectors[i]
                        indices = s_obj.indices.tolist() if hasattr(s_obj, "indices") and hasattr(s_obj.indices, "tolist") else list(s_obj.indices)
                        values = s_obj.values.tolist() if hasattr(s_obj, "values") and hasattr(s_obj.values, "tolist") else list(s_obj.values)
                        vector_dict["sparse"] = SparseVector(indices=indices, values=values)
                        
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector_dict,
                        payload={
                            "text": chunk,
                            "file_name": file_name
                        }
                    ))
                    
                # Отправляем пачку в Qdrant
                self.vector_store.upsert(collection_name=collection_name, points=points)
                
            except Exception as e:
                print(f"❌ Ошибка в пайплайне на батче {start_idx}-{end_idx}: {e}")
                raise
                
            start_idx = end_idx
            
            # Гарантированное уничтожение тяжелых объектов и очистка RAM перед следующим батчем
            del batch_chunks
            del batch_dense_vectors
            del batch_sparse_vectors
            del points
            gc.collect()
        print(f"✅ '{file_name}' успешно загружен и синхронизирован с Qdrant.")
    
    def query_with_contexts(
        self, 
        question: str, 
        collection_name: Optional[str] = None, 
        top_k: Optional[int] = None
    ) -> Tuple[None, List[str], Dict[str, float]]:
        """
        Улучшенный поиск с использованием Reranking, оптимизированный под стриминг Chainlit.
        Возвращает список текстовых контекстов и задержки этапов.
        """
        target_collection = collection_name or self.config.vector_store.collection_name
        actual_top_k = top_k if top_k is not None else self.config.top_k
        
        latencies = {}
        start_total = time.perf_counter()
        
        try:
            # 1. ПОИСК В БД 
            start_step = time.perf_counter()
            
            # Определяем лимит для поиска: если есть реранкер, берем больше для последующей сортировки
            search_limit = self.config.reranker.top_n_retrieval if self.config.reranker.use_reranker else actual_top_k
            
            initial_results = self.vector_store.search(
                query_text=question, 
                collection_name=target_collection,
                limit=search_limit
            )
            latencies['retrieval_time'] = time.perf_counter() - start_step

            # 2. РЕРАНКИРОВАНИЕ (Reranking)
            start_step = time.perf_counter()
            if self.reranker and self.config.reranker.use_reranker and initial_results:
                passages = [res.payload.get("text", "") for res in initial_results if res.payload]
                
                # Вызываем вашу модель реранкера
                reranked_output = self.reranker.rerank(question, passages)
                
                # Безопасная проверка: если реранкер возвращает индексы (int), собираем по индексам.
                # Если он возвращает уже отсортированные строки (str), берем их напрямую.
                if reranked_output and isinstance(reranked_output[0], int):
                    context_list = [passages[idx] for idx in reranked_output[:actual_top_k]]
                else:
                    context_list = reranked_output[:actual_top_k]
                    
                latencies['reranker_time'] = time.perf_counter() - start_step
            else:
                context_list = [res.payload.get("text", "") for res in initial_results if res.payload][:actual_top_k]
                latencies['reranker_time'] = 0.0

            # 3. ФИКСАЦИЯ ИТОГОВЫХ ЗАДЕРЖЕК ПОИСКА
            latencies['context_assembly_time'] = 0.0  # Для совместимости с логами
            latencies['total_time'] = time.perf_counter() - start_total
            
            # Возвращаем None вместо ответа LLM, так как генерация уходит в app.py (Chainlit)
            return None, context_list, latencies

        except Exception as e:
            print(f"❌ Ошибка в query_with_contexts: {e}")
            raise e

    def query(self, question: str, collection_name: Optional[str] = None) -> str:
        """
        Метод-обертка для вызова из консольного main.py.
        Здесь генерация LLM сохранена, чтобы консольный режим не сломался.
        """
        # 1. Извлекаем контексты через наш обновленный метод
        _, context_list, latencies = self.query_with_contexts(question, collection_name)
        
        # 2. КРАСИВЫЙ ВЫВОД НАЙДЕННОГО КОНТЕКСТА В КОНСОЛЬ
        print(f"\n{'='*20} 🔍 НАЙДЕННЫЙ КОНТЕКСТ (Top-{len(context_list)}) {'='*20}")
        if not context_list:
            print("⚠️ Релевантный контекст не найден в базе данных.")
            context = "Информация отсутствует."
        else:
            for idx, text in enumerate(context_list, 1):
                preview_text = text if len(text) <= 400 else f"{text[:400]}..."
                print(f"\n📄 [Фрагмент #{idx}]")
                print(f"{'-'*60}")
                print(preview_text)
                print(f"{'-'*60}")
            context = "\n---\n".join(context_list)
        
        # 3. ЛОКАЛЬНАЯ ГЕНЕРАЦИЯ ДЛЯ КОНСОЛИ
        start_step = time.perf_counter()
        prompt = (
            f"Используй только предоставленный контекст, чтобы ответить на вопрос. "
            f"Если в контексте нет ответа, скажи, что ты не знаешь.\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ВОПРОС:\n{question}\n\n"
            f"ОТВЕТ:"
        )
        response = self.llm.generate(prompt)
        latencies['generation_time'] = time.perf_counter() - start_step
        latencies['total_time'] += latencies['generation_time']
        
        # 4. ВЫВОД МЕТРИК ВРЕМЕНИ В КОНСОЛЬ
        print(f"\n⏱️  [Metrics]")
        print(f"  ├─ Поиск в БД (Retrieval): {latencies.get('retrieval_time', 0):.3f}s")
        print(f"  ├─ Реранкинг (Rerank):     {latencies.get('reranker_time', 0):.3f}s")
        print(f"  ├─ Генерация LLM:          {latencies.get('generation_time', 0):.3f}s")
        print(f"  🚀 Общее время (Total):    {latencies.get('total_time', 0):.3f}s")
        print(f"{'='*70}\n")
        
        return response
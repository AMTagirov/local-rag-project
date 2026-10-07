import os
from src.services.document_parser import PDFDocumentParser
from src.services.text_splitter import ChunkTextSplitter
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import QdrantService
from src.services.llm_service import OllamaLLMService
from src.services.reranker_service import RerankerService 
from src.pipelines.rag_pipeline import RAGPipeline
from src.config.schema import RAGConfig
from src.services.sparse_embedding_service import SparseEmbeddingService

def main():
    # --- 1. КОНФИГУРАЦИЯ ---
    CONFIG_PATH = "src/config/config.yaml"

    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Ошибка: Конфигурация не найдена по пути '{CONFIG_PATH}'")
        return

    print("🚀 Запуск системы RAG...")

    try:
        # Загружаем единый объект конфигурации
        config = RAGConfig.from_yaml(CONFIG_PATH)
        
        collection_name = config.vector_store.collection_name
        docs_dir = config.vector_store.docs_dir

        # --- 2. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ ---
        parser = PDFDocumentParser()
        
        # Шаг А: Сначала создаем модель эмбеддингов
        embedder = EmbeddingService(config.embedding)
        
        # Шаг Б: Теперь передаем её в сплиттер для умного чанкинга
        # ИСПРАВЛЕНО: Добавлен аргумент embedder=embedder
        splitter = ChunkTextSplitter(config.splitter, embedder=embedder)
        
        sparse_embedder = SparseEmbeddingService() 
        vector_db = QdrantService(config.vector_store, embedder=embedder)
        llm = OllamaLLMService(config.llm)
        reranker = RerankerService(config.reranker)

        # --- 3. СБОРКА ПАЙПЛАЙНА ---
        pipeline = RAGPipeline(
            config=config,
            parser=parser,
            splitter=splitter,
            embedder=embedder,
            vector_store=vector_db,
            llm=llm,
            reranker=reranker,
            sparse_service=sparse_embedder
        )

        # --- 4. РАБОТА ---
        print(f"📂 Шаг 1: Синхронизация папки '{docs_dir}' с базой данных...")
        pipeline.sync_directory()

        print("\n✨ Система готова! Введите ваш вопрос или нажмите Enter для теста.")
        test_question = "В чем основная суть документов?"
        
        while True:
            user_input = input("\n❓ Ваш вопрос (или 'exit' для выхода): ").strip()
            if user_input.lower() in ['exit', 'quit', 'выход']:
                break
            if not user_input:
                user_input = test_question
            
            print("🤖 Думаю...")
            answer = pipeline.query(user_input, collection_name)
            
            print(f"\n\n✅ ОТВЕТ:\n{answer}")
            print("-" * 30)

    except Exception as e:
        print(f"❌ Произошла критическая ошибка: {e}")

if __name__ == "__main__":
    main()
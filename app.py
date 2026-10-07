import os
import chainlit as cl  # <-- УБЕДИТЕСЬ, ЧТО ЭТА СТРОКА СТОИТ ТУТ!
import time

# Импорт вашей RAG-архитектуры
from src.config.schema import RAGConfig
from src.services.document_parser import PDFDocumentParser
from src.services.text_splitter import ChunkTextSplitter
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import QdrantService
from src.services.sparse_embedding_service import SparseEmbeddingService
from src.services.llm_service import OllamaLLMService
from src.services.reranker_service import RerankerService 
from src.pipelines.rag_pipeline import RAGPipeline

CONFIG_PATH = "src/config/config.yaml"

@cl.on_chat_start
async def start():
    """Вызывается один раз при старте сессии пользователя."""
    if not os.path.exists(CONFIG_PATH):
        await cl.Message(content=f"❌ Ошибка: Конфиг не найден по пути '{CONFIG_PATH}'").send()
        return

    # 1. Загрузка конфигурации
    config = RAGConfig.from_yaml(CONFIG_PATH)
    collection_name = config.vector_store.collection_name

    # 2. Инициализация всех ваших сервисов
    parser = PDFDocumentParser()
    embedder = EmbeddingService(config.embedding)
    splitter = ChunkTextSplitter(config.splitter, embedder=embedder)
    sparse_embedder = SparseEmbeddingService() 
    vector_db = QdrantService(config.vector_store, embedder=embedder)
    llm = OllamaLLMService(config.llm)
    reranker = RerankerService(config.reranker)

    # 3. Сборка пайплайна
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

    # Сохраняем сервисы в сессию пользователя
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("llm", llm)
    cl.user_session.set("collection_name", collection_name)

    await cl.Message(content="🚀 **Локальная RAG-система успешно запущена!** Задайте ваш вопрос.").send()


@cl.on_message
async def main(message: cl.Message):
    """Вызывается при отправке каждого сообщения пользователем."""
    pipeline: RAGPipeline = cl.user_session.get("pipeline")
    llm: OllamaLLMService = cl.user_session.get("llm")
    collection_name: str = cl.user_session.get("collection_name")

    # 1. ПОЛУЧАЕМ КОНТЕКСТЫ ИЗ ВЕКТОРНОЙ БАЗЫ ДАННЫХ И РЕРАНКЕРА
    _, context_list, latencies = await cl.make_async(pipeline.query_with_contexts)(
        message.content, collection_name
    )

    # 2. КРАСИВОЕ ФОРМАТИРОВАНИЕ И ОЧИСТКА СТАРЫХ ЧАНКОВ
    text_elements = []
    if context_list:
        formatted_side_panel = ""
        for idx, text in enumerate(context_list, 1):
            # Тройные кавычки делают шрифт мелким, моноширинным и аккуратным
            formatted_side_panel += (
                f"### 📄 Фрагмент #{idx}\n"
                f"```text\n"
                f"{text}\n"
                f"```\n\n"
                f"---\n\n"
            )
        
        element_name = "Контекст запроса"
        text_elements.append(
            cl.Text(
                name=element_name,
                content=formatted_side_panel,
                display="side"  
            )
        )

    # 3. ФОРМИРУЕМ ПРОМПТ
    context_str = "\n\n---\n\n".join(context_list) if context_list else "Информация отсутствует."
    prompt = (
        "Используй только предоставленный текст контекста, чтобы ответить на вопрос.\n"
        "Если в контексте нет ответа, честно скажи, что ты не знаешь.\n\n"
        f"КОНТЕКСТ:\n{context_str}\n\n"
        f"ВОПРОС: {message.content}\n\n"
        "ОТВЕТ:"
    )

    # 4. ЗАПУСКАЕМ ПОТОКОВЫЙ ВЫВОД (STREAMING) С ПОДСЧЕТОМ ЛАТЕНТНОСТИ
    msg = cl.Message(content="", elements=text_elements)
    await msg.send()

    try:
        # Фиксируем время СТАРТА генерации LLM
        start_llm = time.perf_counter()
        
        # Получаем асинхронный поток токенов из Ollama
        async_stream = await cl.make_async(llm.generate_stream)(prompt)
        
        for token in async_stream:
            await msg.stream_token(token)
            
        # Фиксируем время ОКОНЧАНИЯ генерации LLM
        llm_latency = time.perf_counter() - start_llm

        # 5. ВЫВОД ПОЛНЫХ МЕТРИК ВРЕМЕНИ
        # retrieval_time и reranker_time мы по-прежнему забираем из словаря latencies пайплайна,
        # а llm_latency подставляем из нашего локального замера.
        metrics_str = (
            f"\n\n---\n*⏱️ Время этапов: "
            f"Retrieval (Гибрид): {latencies.get('retrieval_time', 0):.2f}s | "
            f"Rerank: {latencies.get('reranker_time', 0):.2f}s | "
            f"LLM (Streaming): {llm_latency:.2f}s*"
        )
        
        await msg.stream_token(metrics_str)
        await msg.update()

    except Exception as e:
        msg.content = f"❌ Произошла ошибка во время генерации ответа: {e}"
        await msg.update()
import os
import threading
import time

import chainlit as cl

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
_system = None
_system_lock = threading.Lock()


def initialize_system():
    """Один раз загружает тяжёлые модели и возвращает общий пайплайн."""
    global _system

    if _system is not None:
        return _system

    with _system_lock:
        if _system is not None:
            return _system

        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Конфиг не найден по пути '{CONFIG_PATH}'")

        config = RAGConfig.from_yaml(CONFIG_PATH)
        parser = PDFDocumentParser()
        embedder = EmbeddingService(config.embedding)
        splitter = ChunkTextSplitter(config.splitter, embedder=embedder)
        sparse_embedder = SparseEmbeddingService()
        vector_db = QdrantService(config.vector_store, embedder=embedder)
        llm = OllamaLLMService(config.llm)
        reranker = RerankerService(config.reranker)

        pipeline = RAGPipeline(
            config=config,
            parser=parser,
            splitter=splitter,
            embedder=embedder,
            vector_store=vector_db,
            llm=llm,
            reranker=reranker,
            sparse_service=sparse_embedder,
        )
        _system = (pipeline, llm, config.vector_store.collection_name)
        return _system

@cl.on_chat_start
async def start():
    """Вызывается один раз при старте сессии пользователя."""
    status = cl.Message(
        content="⏳ **Загрузка RAG-системы...** Подготавливаю модели и подключение к Qdrant."
    )
    await status.send()

    try:
        pipeline, llm, collection_name = await cl.make_async(initialize_system)()
    except Exception as error:
        status.content = f"❌ **Не удалось загрузить RAG-систему:** {error}"
        await status.update()
        return

    # Сохраняем сервисы в сессию пользователя
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("llm", llm)
    cl.user_session.set("collection_name", collection_name)

    status.content = "🚀 **Локальная RAG-система готова!** Задайте ваш вопрос."
    await status.update()


@cl.on_message
async def main(message: cl.Message):
    """Вызывается при отправке каждого сообщения пользователем."""
    pipeline: RAGPipeline = cl.user_session.get("pipeline")
    llm: OllamaLLMService = cl.user_session.get("llm")
    collection_name: str = cl.user_session.get("collection_name")

    if pipeline is None or llm is None or collection_name is None:
        await cl.Message(
            content="⏳ RAG-система ещё не готова. Дождитесь сообщения о завершении загрузки."
        ).send()
        return

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
    prompt = pipeline.build_prompt(message.content, context_list)

    # 4. ЗАПУСКАЕМ ПОТОКОВЫЙ ВЫВОД (STREAMING) С ПОДСЧЕТОМ ЛАТЕНТНОСТИ
    msg = cl.Message(content="", elements=text_elements)
    await msg.send()

    try:
        # Фиксируем время СТАРТА генерации LLM
        start_llm = time.perf_counter()
        
        # Асинхронно читаем токены, не блокируя event loop Chainlit.
        async for token in llm.generate_stream(prompt):
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

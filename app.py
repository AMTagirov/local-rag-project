import asyncio
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict

import chainlit as cl

# Импорт вашей RAG-архитектуры
from src.config.schema import RAGConfig
from src.services.document_parser import create_document_parser
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
_ingestion_lock = threading.Lock()
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx"}


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
        parser = create_document_parser(config)
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


async def update_documents_sidebar(
    pipeline: RAGPipeline,
    collection_name: str,
) -> None:
    """Показывает актуальный список документов в отдельной боковой панели."""
    file_names = await cl.make_async(
        pipeline.vector_store.get_existing_file_names
    )(collection_name)
    ordered_names = sorted(file_names, key=str.casefold)
    if ordered_names:
        content = "\n".join(
            f"{index}. 📄 `{file_name}`"
            for index, file_name in enumerate(ordered_names, start=1)
        )
    else:
        content = "_Коллекция пока не содержит документов._"

    await cl.ElementSidebar.set_title("Документы коллекции")
    await cl.ElementSidebar.set_elements([
        cl.Text(
            name="Список документов",
            content=(
                f"**Коллекция:** `{collection_name}`  \n"
                f"**Документов:** {len(ordered_names)}\n\n{content}"
            ),
            display="side",
        )
    ])


def save_and_ingest_uploaded_file(
    uploaded_file: Any,
    pipeline: RAGPipeline,
    collection_name: str,
    progress_callback,
) -> Dict[str, Any]:
    """Сохраняет файл в data/docs и индексирует его под общей блокировкой."""
    safe_name = Path(uploaded_file.name).name
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(f"Файл '{safe_name}': поддерживаются только PDF и DOCX")

    docs_dir = Path(pipeline.config.vector_store.docs_dir)
    destination = docs_dir / safe_name

    with _ingestion_lock:
        existing_files = pipeline.vector_store.get_existing_file_names(collection_name)
        if safe_name in existing_files:
            raise FileExistsError(f"Документ '{safe_name}' уже находится в коллекции")
        if destination.exists():
            raise FileExistsError(f"Файл '{safe_name}' уже существует в {docs_dir}")

        docs_dir.mkdir(parents=True, exist_ok=True)
        if uploaded_file.path:
            shutil.copy2(uploaded_file.path, destination)
        elif isinstance(uploaded_file.content, (bytes, bytearray)):
            destination.write_bytes(uploaded_file.content)
        else:
            raise ValueError(f"Не удалось прочитать загруженный файл '{safe_name}'")

        try:
            return pipeline.ingest_document(
                str(destination),
                collection_name,
                progress_callback=progress_callback,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise


def format_ingestion_progress(file_name: str, event: Dict[str, Any]) -> str:
    stage = event.get("stage")
    chunks_added = event.get("chunks_added", 0)
    if stage == "parsing":
        detail = "чтение и очистка документа"
    elif stage == "embedding":
        detail = f"чанкинг и векторизация; обработано чанков: {chunks_added}"
    elif stage == "indexing":
        detail = f"запись в Qdrant; добавлено чанков: {chunks_added}"
    elif stage == "complete":
        detail = f"индексация завершена; добавлено чанков: {chunks_added}"
    else:
        detail = "ожидание индексации"
    return f"⏳ **{file_name}**\n\n{detail}"


async def ingest_uploaded_file(
    uploaded_file: Any,
    pipeline: RAGPipeline,
    collection_name: str,
) -> bool:
    """Индексирует один элемент Chainlit и обновляет прогресс в сообщении."""
    file_name = Path(uploaded_file.name).name
    progress_message = cl.Message(
        content=f"⏳ **{file_name}**\n\nФайл поставлен в очередь на индексацию."
    )
    await progress_message.send()

    loop = asyncio.get_running_loop()
    progress_queue: asyncio.Queue = asyncio.Queue()

    def report_progress(event: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(progress_queue.put_nowait, event)

    task = asyncio.create_task(
        cl.make_async(save_and_ingest_uploaded_file)(
            uploaded_file,
            pipeline,
            collection_name,
            report_progress,
        )
    )

    try:
        while not task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            progress_message.content = format_ingestion_progress(file_name, event)
            await progress_message.update()

        result = await task
        while not progress_queue.empty():
            event = progress_queue.get_nowait()
            progress_message.content = format_ingestion_progress(file_name, event)

        progress_message.content = (
            f"✅ **{result['file_name']}** добавлен в коллекцию "
            f"`{result['collection_name']}`.\n\n"
            f"Создано чанков: **{result['chunks_added']}**."
        )
        await progress_message.update()
        return True
    except Exception as error:
        progress_message.content = f"❌ **Не удалось загрузить {file_name}:** {error}"
        await progress_message.update()
        return False

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

    await update_documents_sidebar(pipeline, collection_name)
    status.content = (
        "🚀 **Локальная RAG-система готова!** Задайте вопрос или перетащите "
        "PDF/DOCX в окно чата, чтобы добавить документ."
    )
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

    uploaded_documents = [
        element
        for element in message.elements
        if Path(element.name).suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]
    if message.elements and not uploaded_documents:
        await cl.Message(content="❌ Поддерживаются только документы PDF и DOCX.").send()
        return

    if uploaded_documents:
        any_success = False
        for uploaded_file in uploaded_documents:
            any_success |= await ingest_uploaded_file(
                uploaded_file,
                pipeline,
                collection_name,
            )
        if any_success:
            await update_documents_sidebar(pipeline, collection_name)
        if not message.content.strip():
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
        async for token in llm.generate_stream(
            prompt,
            system_prompt=pipeline.RAG_SYSTEM_PROMPT,
        ):
            await msg.stream_token(token)
            
        # Фиксируем время ОКОНЧАНИЯ генерации LLM
        llm_latency = time.perf_counter() - start_llm

        # 5. ВЫВОД ПОЛНЫХ МЕТРИК ВРЕМЕНИ
        # retrieval_time и reranker_time мы по-прежнему забираем из словаря latencies пайплайна,
        # а llm_latency подставляем из нашего локального замера.
        metrics_str = (
            f"\n\n---\n*⏱️ Время этапов: "
            f"Rewrite: {latencies.get('query_rewrite_time', 0):.2f}s | "
            f"Retrieval (Гибрид): {latencies.get('retrieval_time', 0):.2f}s | "
            f"Rerank: {latencies.get('reranker_time', 0):.2f}s | "
            f"LLM (Streaming): {llm_latency:.2f}s*"
        )
        
        await msg.stream_token(metrics_str)
        await msg.update()

    except Exception as e:
        msg.content = f"❌ Произошла ошибка во время генерации ответа: {e}"
        await msg.update()

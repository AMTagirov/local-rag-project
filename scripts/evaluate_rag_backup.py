import json
import os
import sys
from types import ModuleType
from datasets import Dataset

# 1. Обход устаревшего импорта в старых версиях ragas (если применимо)
try:
    import langchain_community.chat_models.vertexai
except ImportError:
    mock_module = ModuleType("langchain_community.chat_models.vertexai")
    mock_module.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = mock_module


from ragas import evaluate

from openai import OpenAI as OpenAIClient
from ragas.llms import llm_factory

# В современных версиях Ragas используется LangchainLLMWrapper или непосредственно Langchain-модели
from ragas.llms import LangchainLLMWrapper 
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, SemanticSimilarity
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama

# Предполагаемые импорты ваших локальных сервисов

from src.services.document_parser import PDFDocumentParser
from src.services.text_splitter import ChunkTextSplitter
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import QdrantService
from src.services.llm_service import OllamaLLMService
from src.pipelines.rag_pipeline import RAGPipeline


def get_rag_predictions(pipeline: RAGPipeline, question: str, collection: str) -> tuple[str, list[str], dict]:
    """Возвращает сгенерированный ответ, список контекстов и словарь задержек."""
    try:
        # Теперь получаем три значения, так как pipeline.query_with_contexts изменился
        answer, retrieved_contexts, latencies = pipeline.query_with_contexts(question, collection)
        return answer, retrieved_contexts, latencies
    except Exception as e:
        print(f"❌ Ошибка на вопросе {question[:30]}... : {e}")
        return "", [], {}

def run_evaluation(dataset_path: str, collection_name: str, llm_name: str):
    import mlflow  # Импорт внутри функции для гибкости
    
    if not os.path.exists(dataset_path):
        print(f"❌ Файл {dataset_path} не найден.")
        return

    # Инициализируем MLflow
    mlflow.set_experiment("RAG_Performance_Optimization")
    
    with mlflow.start_run(run_name=f"eval_{llm_name}"):
        # Логируем основные параметры
        mlflow.log_param("llm_model", llm_name)
        mlflow.log_param("collection", collection_name)
    
    # 1. Инициализация RAG пайплайна
    pipeline = RAGPipeline(
        PDFDocumentParser(),
        ChunkTextSplitter(chunk_size=500, chunk_overlap=50),
        EmbeddingService("intfloat/multilingual-e5-large"),
        QdrantService(host="localhost", port=6333),
        OllamaLLMService(model_name=llm_name)
    )

    # 2. Загрузка данных
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    print(f"🧪 Тестирование на {len(golden_data)} примерах...")

    # 3. Сбор предсказаний
    questions = []
    references = []
    generated_answers = []
    retrieved_contexts_list = []
    all_latencies = [] # Список для сбора всех замеров времени
    all_row_results = [] # Здесь будем хранить DataFrame для каждой строки


    for i, item in enumerate(golden_data):
        q = item["question"]
        gt = item["ground_truth"]
        if isinstance(gt, list):
            reference_str = gt[0] if gt else ""
        else:
            reference_str = str(gt)

        print(f"   [Row {i+1}/{len(golden_data)}] Обработка вопроса: {q[:50]}...")

        # 1. Получаем ответ и задержки (Один запрос к LLM за раз)
        ans, contexts, latencies = get_rag_predictions(pipeline, q, collection_name)
        
        if latencies:
            all_latencies.append(latencies)

        # 2. Создаем "микро-датасет" из ОДНОЙ строки для Ragas
        # Это заставляет Ragas работать в один поток для этого вопроса
        tiny_dataset = Dataset.from_dict({
            "question": [q],
            "contexts": [contexts],
            "reference": [reference_str],
            "answer": [ans]
        })

        # 3. Оцениваем только этот один вопрос
        # Настройка судей (LLM и Embedding) остается такой же
        print("🤖 Инициализация современных моделей-судей (Ragas InstructorLLM)...")
        
        # Настраиваем клиент OpenAI на работу с локальным сервером Ollama
        openai_client = OpenAIClient(
            base_url=f"{config.llm.base_url}/v1",  # Добавляем суффикс /v1 для OpenAI-совместимости Ollama
            api_key="ollama"  # Ollama принимает любую строку в качестве API-ключа
        )
        
        # Создаем InstructorLLM судью, которого требует новая версия Ragas
        judge_llm = llm_factory(
            model=config.llm.model_name,
            client=openai_client
        )
        
        # Судья-Embeddings (оставляем старый рабочий вариант, если он не вызывает ошибок)
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        judge_emb = LangchainEmbeddingsWrapper(hf_embeddings)
        
        # Передаем новый judge_llm во все коллекции метрик
        metrics = [
            Faithfulness(llm=judge_llm),
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
            SemanticSimilarity(embeddings=judge_emb)
        ]

        # Запускаем оценку для одного вопроса
        result = evaluate(dataset=tiny_dataset, metrics=metrics)
        
        # Сохраняем DataFrame этого одного вопроса для последующего усреднения
        all_row_results.append(result.to_pandas())

    # 4. Агрегация результатов (Объединяем все маленькие DataFrame в один большой)
    print("📊 Собираю и усредняю результаты...")
    final_df = pd.concat(all_row_results, ignore_index=True)
    final_metrics_dict = final_df.mean().to_dict()


    print("\n📊 Результаты оценки:")
    print(final_metrics_dict)

     # --- Логирование в MLflow ---
    
    # 1. Логируем метрики Ragas (преобразуем pandas Series в dict)
    
    mlflow.log_metrics(final_metrics_dict)

    # 2. Логируем средние задержки (Latency)
    if all_latencies:
        avg_latencies = {}
        # Берем ключи из первого успешного замера
        keys = all_latencies[0].keys()
        for key in keys:
            avg_latencies[key] = sum(l.get(key, 0) for l in all_latencies) / len(all_latencies)
        
        print(f"⏱ Средние задержки: {avg_latencies}")
        mlflow.log_metrics(avg_latencies)
    else:
        print("⚠️ Не удалось собрать данные о задержках.")
    
if __name__ == "__main__":
    run_evaluation("golden_dataset.json", "my_knowledge_base", "qwen2.5:14b")

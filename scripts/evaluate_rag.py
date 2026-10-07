import json
import os
import sys
import time
from types import ModuleType
from typing import List, Dict, Any, Tuple

import pandas as pd
from datasets import Dataset

# 1. Обход устаревшего импорта (оставляем для совместимости сред)
try:
    import langchain_community.chat_models.vertexai
except ImportError:
    mock_module = ModuleType("langchain_community.chat_models.vertexai")
    mock_module.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = mock_module

from openai import OpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper 
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, SemanticSimilarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Ваши сервисы
from src.config.schema import RAGConfig
from src.services.document_parser import PDFDocumentParser  
from src.services.text_splitter import ChunkTextSplitter
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import QdrantService
from src.services.sparse_embedding_service import SparseEmbeddingService
from src.services.llm_service import OllamaLLMService
from src.services.reranker_service import RerankerService 
from src.pipelines.rag_pipeline import RAGPipeline


def get_rag_predictions(
    pipeline: RAGPipeline, 
    question: str, 
    collection_name: str
) -> Tuple[str, List[str], Dict[str, float]]:
    """Возвращает сгенерированный ответ, список контекстов и словарь задержек."""
    try:
        return pipeline.query_with_contexts(question, collection_name)
    except Exception as e:
        print(f"❌ Ошибка на вопросе '{question[:30]}...': {e}")
        return "", [], {}

def run_evaluation(config: RAGConfig, dataset_path: str):
    """
    Запускает процесс оценки RAG системы с использованием Ragas и MLflow.
    """
    import mlflow 
    from mlflow.tracking import MlflowClient
    
    if not os.path.exists(dataset_path):
        print(f"❌ Файл датасета {dataset_path} не найден.")
        return

    client = MlflowClient()
    exp_name = config.experiment_name

    # Проверяем, существует ли эксперимент (включая удаленные)
    existing_exp = client.get_experiment_by_name(exp_name)

    if existing_exp:
        if existing_exp.lifecycle_stage == "deleted":
            print(f"Эксперимент '{exp_name}' удален. Восстанавливаем...")
            client.restore_experiment(existing_exp.experiment_id)
    else:
        client.create_experiment(exp_name)

    mlflow.set_experiment(exp_name)
    
    # Имя коллекции берем из конфига
    collection_name = config.vector_store.collection_name
    llm_model_name = config.llm.model_name

    # Гарантированная инициализация парсера на основе конфига
    if config.parser_type == "pdf":
        parser = PDFDocumentParser()
    else:
        parser = PDFDocumentParser() # Фолбэк-заглушка

    print(f"🧪 Запуск оценки для модели {llm_model_name} на коллекции '{collection_name}'")
    
    with mlflow.start_run(run_name=f"eval_{llm_model_name.replace(':', '_')}"):
        mlflow.log_param("llm_model", llm_model_name)
        mlflow.log_param("collection", collection_name)
        mlflow.log_param("parser_type", config.parser_type) 
        
        # --- 1. ИНИЦИАЛИЗАЦИЯ ПАЙПЛАЙНА (ОПТИМИЗИРОВАНО: Сквозные зависимости интерфейсов) ---
        embedder = EmbeddingService(config.embedding)
        splitter = ChunkTextSplitter(config.splitter, embedder=embedder)
        vector_db = QdrantService(config.vector_store, embedder=embedder)
        sparse_embedder = SparseEmbeddingService() 
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
            sparse_service=sparse_embedder
        )

        # --- 2. ЗАГРУЗКА ЗОЛОТОГО ДАТАСЕТА ---
        with open(dataset_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
        print(f"📖 Загружено {len(golden_data)} примеров из датасета.")

        # --- 3. ИНИЦИАЛИЗАЦИЯ СУДЕЙ (Сохранено без изменений) ---
        print("🤖 Инициализация моделей-судей (Ragas)...")
        
        native_judge_llm = ChatOllama(
            model=config.llm.model_name, 
            base_url=config.llm.base_url, 
            num_ctx=config.llm.num_ctx,
            timeout=300.0
        )
        judge_llm = LangchainLLMWrapper(native_judge_llm)
        
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        judge_emb = LangchainEmbeddingsWrapper(hf_embeddings)
        
        metrics = [
            Faithfulness(llm=judge_llm),
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
            SemanticSimilarity(embeddings=judge_emb)
        ]

        all_latencies = [] 
        evaluation_rows = [] 

        # --- 4. ОПТИМИЗИРОВАННЫЙ ЦИКЛ СБОРА ОТВЕТОВ ---
        for i, item in enumerate(golden_data):
            q = item["question"]
            gt = item["ground_truth"]
            reference_str = gt[0] if isinstance(gt, list) else str(gt)

            print(f"   [Строка {i+1}/{len(golden_data)}] Генерация ответа RAG пайплайном...")
            ans, contexts, latencies = get_rag_predictions(pipeline, q, collection_name)
            
            if latencies:
                all_latencies.append(latencies)

            # Накапливаем строки для создания единого датасета
            evaluation_rows.append({
                "question": q,
                "contexts": contexts,
                "reference": reference_str,
                "answer": ans
            })

        # Создаем финальный датасет для Ragas
        ragas_dataset = Dataset.from_pandas(pd.DataFrame(evaluation_rows))

        print("📊 Запуск глобального анализа качества через Ragas (Линейный режим)...")
        try:
            # Настраиваем конфигурацию запуска Ragas: 
            # Задаем max_workers=1 и таймаут внутри RunConfig, который поддерживается всеми версиями
            from ragas.run_config import RunConfig
            
            custom_run_config = RunConfig(
                max_workers=4,  
                timeout=300     # 5 минут на сложный запрос-анализ
            )
            
            # Вызываем evaluate с явной конфигурацией выполнения
            result = evaluate(
                dataset=ragas_dataset, 
                metrics=metrics,
                run_config=custom_run_config 
            )
            
            final_df = result.to_pandas()
            
        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ОЦЕНКИ RAGAS: {e}")
            return
        # --- 5. АГРЕГАЦИЯ И LOGGING ---
        if final_df.empty:
            print("❌ Не удалось собрать результаты оценки.")
            return

        print("📊 Сборка финальных результатов...")
        
        numeric_cols = final_df.select_dtypes(include=['number'])
        final_metrics_dict = numeric_cols.mean().to_dict()
        
        if "contexts" in final_df.columns:
            final_df["contexts_formatted"] = final_df["contexts"].apply(
                lambda c: "\n\n".join([f"[{idx+1}] {text}" for idx, text in enumerate(c)]) if isinstance(c, list) else str(c)
            )

        readable_cols = []
        for col in ["question", "reference", "answer", "contexts_formatted"]:
            if col in final_df.columns:
                readable_cols.append(col)
        
        other_cols = [c for c in final_df.columns if c not in readable_cols and c != "contexts"]
        report_df = final_df[readable_cols + other_cols]

        report_csv_path = "evaluation_report_detailed.csv"
        report_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")
        
        report_xlsx_path = "evaluation_report_detailed.xlsx"
        try:
            report_df.to_excel(report_xlsx_path, index=False, engine='xlsxwriter')
        except ImportError:
            print("⚠️ Библиотека 'xlsxwriter' не найдена. Сохраняю стандартным движком...")
            report_df.to_excel(report_xlsx_path, index=False)
        
        mlflow.log_artifact(report_csv_path)
        mlflow.log_artifact(report_xlsx_path)
        print(f"📄 Детальный отчет сохранен в: {report_csv_path} и {report_xlsx_path}")

        print("📥 Отправка интерактивной таблицы результатов в MLflow...")
        mlflow.log_table(data=report_df, artifact_file="evaluation_summary_table.json")

        # --- 6. ОБРАБОТКА ЗАДЕРЖЕК (LATENCY) ---
        avg_latencies = {}
        if all_latencies:
            print("⏱️ Вычисление средних задержек...")
            # ИСПРАВЛЕНО: Безопасное извлечение ключей из первого словаря в списке
            latency_keys = all_latencies[0].keys()
            for key in latency_keys:
                values = [lat[key] for lat in all_latencies if key in lat]
                if values:
                    avg_latencies[f"latency_{key}"] = sum(values) / len(values)
            
            mlflow.log_metrics(avg_latencies)

        # --- 7. ФИНАЛЬНЫЙ ВЫВОД ---
        print("\n🏆 Итоговые метрики Ragas:")
        for k, v in final_metrics_dict.items():
            print(f"   {k}: {v:.4f}")

        if avg_latencies:
            print("\n⏱️ Среднее время выполнения этапов (сек):")
            for k, v in avg_latencies.items():
                print(f"   {k.replace('latency_', '')}: {v:.4f}s")

        mlflow.log_metrics(final_metrics_dict)
        
    print("\n✅ Оценка успешно завершена! Всё залогировано в MLflow.")


if __name__ == "__main__":
    CONFIG_FILE = "src/config/config.yaml"
    DATASET_FILE = "golden_dataset.json"

    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Конфиг не найден: {CONFIG_FILE}")
    elif not os.path.exists(DATASET_FILE):
        print(f"❌ Датасет не найден: {DATASET_FILE}")
    else:
        app_config = RAGConfig.from_yaml(CONFIG_FILE)
        run_evaluation(app_config, DATASET_FILE)
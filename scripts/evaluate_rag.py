import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.services.document_parser import create_document_parser
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
) -> Tuple[str, List[str], List[Dict[str, Any]], Dict[str, float]]:
    """Возвращает ответ, контексты, их ID и задержки."""
    try:
        context_records, latencies = pipeline.query_with_context_records(
            question, collection_name
        )
        contexts = [record["text"] for record in context_records]
        retrieved_sources = [
            {
                "file_name": record.get("file_name"),
                "chunk_index": record.get("chunk_index"),
            }
            for record in context_records
        ]

        start_generation = time.perf_counter()
        prompt = pipeline.build_prompt(question, contexts)
        answer = pipeline.llm.generate(
            prompt,
            system_prompt=pipeline.RAG_SYSTEM_PROMPT,
        )
        generation_time = time.perf_counter() - start_generation

        latencies["generation_time"] = generation_time
        latencies["total_time"] = latencies.get("total_time", 0.0) + generation_time

        return answer, contexts, retrieved_sources, latencies
    except Exception as e:
        print(f"❌ Ошибка на вопросе '{question[:30]}...': {e}")
        return "", [], [], {}


def calculate_id_retrieval_metrics(
    target_source: Dict[str, Any],
    retrieved_sources: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Проверяет попадание единственного золотого чанка в выдачу."""
    def source_key(source: Dict[str, Any]):
        file_name = source.get("file_name")
        chunk_index = source.get("chunk_index")
        if file_name is None or chunk_index is None:
            return None
        return str(file_name), int(chunk_index)

    target_key = source_key(target_source)
    retrieved_keys = [
        key for source in retrieved_sources if (key := source_key(source)) is not None
    ]

    if target_key is None:
        return {
            "id_target_hit_at_k": float("nan"),
            "id_target_mrr": float("nan"),
        }

    target_rank = next(
        (rank for rank, key in enumerate(retrieved_keys, start=1) if key == target_key),
        None,
    )
    return {
        "id_target_hit_at_k": float(target_rank is not None),
        "id_target_mrr": 1.0 / target_rank if target_rank is not None else 0.0,
    }


def load_single_chunk_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Загружает датасет и проверяет, что у примера ровно один золотой чанк."""
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("Золотой датасет должен быть непустым JSON-массивом")

    errors = []
    for row_number, item in enumerate(data, start=1):
        contexts = item.get("contexts")
        source = item.get("source")
        if not item.get("question") or not item.get("ground_truth"):
            errors.append(f"строка {row_number}: нет question или ground_truth")
        if (
            not isinstance(contexts, list)
            or len(contexts) != 1
            or not isinstance(contexts[0], str)
            or not contexts[0].strip()
        ):
            errors.append(f"строка {row_number}: contexts должен содержать ровно один текст")
        if not isinstance(source, dict) or source.get("file_name") is None or source.get("chunk_index") is None:
            errors.append(f"строка {row_number}: нет source.file_name/chunk_index")

    if errors:
        preview = "; ".join(errors[:5])
        suffix = f"; ещё ошибок: {len(errors) - 5}" if len(errors) > 5 else ""
        raise ValueError(
            "Датасет не соответствует схеме одного золотого чанка: "
            f"{preview}{suffix}. Пересоздайте golden dataset."
        )
    return data

def run_evaluation(
    config: RAGConfig,
    dataset_path: str,
    run_name: str | None = None,
    output_dir: str = "evaluation_reports",
    metrics_output: str | None = None,
) -> Dict[str, float] | None:
    """
    Запускает процесс оценки RAG системы с использованием Ragas и MLflow.
    """
    import mlflow 
    from mlflow.tracking import MlflowClient
    
    if not os.path.exists(dataset_path):
        print(f"❌ Файл датасета {dataset_path} не найден.")
        return

    try:
        golden_data = load_single_chunk_dataset(dataset_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"❌ Ошибка золотого датасета: {error}")
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
    effective_run_name = run_name or f"eval_{llm_model_name.replace(':', '_')}"
    run_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", effective_run_name).strip("_")
    report_dir = Path(output_dir) / run_slug
    report_dir.mkdir(parents=True, exist_ok=True)

    # Гарантированная инициализация парсера на основе конфига
    # Оценка не парсит документы; PDF-парсер используется как базовая зависимость
    # пайплайна, а реальная индексация в режиме auto выбирает парсер по расширению.
    parser = create_document_parser(config)

    print(f"🧪 Запуск оценки для модели {llm_model_name} на коллекции '{collection_name}'")
    
    with mlflow.start_run(run_name=effective_run_name):
        mlflow.log_params({
            "llm_model": llm_model_name,
            "collection": collection_name,
            "parser_type": config.parser_type.value,
            "embedding_model": config.embedding.model_name,
            "chunking_strategy": config.splitter.strategy.value,
            "chunk_size": config.splitter.chunk_size,
            "chunk_min_size": config.splitter.chunk_min_size,
            "chunk_overlap": config.splitter.chunk_overlap,
            "search_mode": config.vector_store.search_mode.value,
            "top_k": config.top_k,
            "use_reranker": config.reranker.use_reranker,
            "reranker_model": config.reranker.model_name,
            "top_n_retrieval": config.reranker.top_n_retrieval,
            "query_rewriting": config.query_rewriting.enabled,
            "query_rewriting_keep_original": config.query_rewriting.keep_original,
            "query_rewriting_rrf_k": config.query_rewriting.rrf_k,
            "document_analysis": config.document_analysis.enabled,
            "formula_recognition": config.document_analysis.use_formula_recognition,
            "table_recognition": config.document_analysis.use_table_recognition,
            "chart_recognition": config.document_analysis.use_chart_recognition,
        })
        
        # --- 1. ИНИЦИАЛИЗАЦИЯ ПАЙПЛАЙНА (ОПТИМИЗИРОВАНО: Сквозные зависимости интерфейсов) ---
        embedder = EmbeddingService(config.embedding)
        splitter = ChunkTextSplitter(config.splitter, embedder=embedder)
        vector_db = QdrantService(config.vector_store, embedder=embedder)
        sparse_embedder = SparseEmbeddingService() 
        llm = OllamaLLMService(config.llm)
        reranker = (
            RerankerService(config.reranker)
            if config.reranker.use_reranker
            else None
        )

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
        id_metric_rows = []

        # --- 4. ОПТИМИЗИРОВАННЫЙ ЦИКЛ СБОРА ОТВЕТОВ ---
        for i, item in enumerate(golden_data):
            q = item["question"]
            gt = item["ground_truth"]
            reference_str = gt[0] if isinstance(gt, list) else str(gt)

            print(f"   [Строка {i+1}/{len(golden_data)}] Генерация ответа RAG пайплайном...")
            ans, contexts, retrieved_sources, latencies = get_rag_predictions(
                pipeline, q, collection_name
            )

            if not ans.strip():
                print(f"⚠️ Пропуск строки {i+1}: LLM не вернула ответ.")
                continue
            
            if latencies:
                all_latencies.append(latencies)

            # Накапливаем строки для создания единого датасета
            evaluation_rows.append({
                "question": q,
                "contexts": contexts,
                "reference_contexts": item["contexts"],
                "reference": reference_str,
                "answer": ans
            })
            id_metric_rows.append(calculate_id_retrieval_metrics(
                item.get("source", {}),
                retrieved_sources,
            ))

        if not evaluation_rows:
            print("❌ Нет успешно сгенерированных ответов для оценки.")
            return

        if all(
            row["id_target_hit_at_k"] != row["id_target_hit_at_k"]
            for row in id_metric_rows
        ):
            print(
                "⚠️ ID-метрики не будут рассчитаны: в датасете или коллекции "
                "нет file_name/chunk_index. Переиндексируйте коллекцию и пересоздайте датасет."
            )

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

        for metric_name in id_metric_rows[0]:
            final_df[metric_name] = [row[metric_name] for row in id_metric_rows]
        
        numeric_cols = final_df.select_dtypes(include=['number'])
        final_metrics_dict = numeric_cols.mean().dropna().to_dict()
        
        contexts_column = (
            "retrieved_contexts" if "retrieved_contexts" in final_df.columns else "contexts"
        )
        if contexts_column in final_df.columns:
            final_df["contexts_formatted"] = final_df[contexts_column].apply(
                lambda c: "\n\n".join([f"[{idx+1}] {text}" for idx, text in enumerate(c)]) if isinstance(c, list) else str(c)
            )

        readable_cols = []
        for col in ["user_input", "question", "reference", "response", "answer", "contexts_formatted"]:
            if col in final_df.columns:
                readable_cols.append(col)
        
        other_cols = [
            c for c in final_df.columns
            if c not in readable_cols and c not in {"contexts", "retrieved_contexts"}
        ]
        report_df = final_df[readable_cols + other_cols]

        report_csv_path = report_dir / "evaluation_report_detailed.csv"
        report_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")
        
        report_xlsx_path = report_dir / "evaluation_report_detailed.xlsx"
        try:
            report_df.to_excel(report_xlsx_path, index=False, engine='xlsxwriter')
        except ImportError:
            print("⚠️ Библиотека 'xlsxwriter' не найдена. Сохраняю стандартным движком...")
            report_df.to_excel(report_xlsx_path, index=False)
        
        mlflow.log_artifact(str(report_csv_path))
        mlflow.log_artifact(str(report_xlsx_path))
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

        combined_metrics = {**final_metrics_dict, **avg_latencies}
        if metrics_output:
            metrics_path = Path(metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(combined_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        
    print("\n✅ Оценка успешно завершена! Всё залогировано в MLflow.")
    return combined_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка RAG и логирование в MLflow")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dataset", default="golden_dataset.json")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", default="evaluation_reports")
    parser.add_argument("--metrics-output")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.error(f"Конфиг не найден: {args.config}")
    if not os.path.exists(args.dataset):
        parser.error(f"Датасет не найден: {args.dataset}")

    app_config = RAGConfig.from_yaml(args.config)
    metrics = run_evaluation(
        app_config,
        args.dataset,
        run_name=args.run_name,
        output_dir=args.output_dir,
        metrics_output=args.metrics_output,
    )
    if metrics is None:
        raise SystemExit(1)

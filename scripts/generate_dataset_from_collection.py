import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.schema import RAGConfig
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import QdrantService
from src.services.llm_service import OllamaLLMService


def select_stratified_targets(
    records_by_file: Dict[str, List[Dict[str, Any]]],
    num_questions: int,
    seed: int = 42,
    num_strata: int = 3,
) -> List[Dict[str, Any]]:
    """Равномерно выбирает чанки из разных частей каждого документа."""
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for file_name, records in sorted(records_by_file.items()):
        ordered = sorted(records, key=lambda record: record["chunk_index"])
        for position, record in enumerate(ordered):
            stratum = min(num_strata - 1, position * num_strata // len(ordered))
            buckets[(stratum, file_name)].append(record)

    file_names = sorted(records_by_file)
    bucket_order = []
    seen_keys = set()
    # Чередуем и части документа, и файлы. Даже маленькая выборка
    # не должна целиком попадать только в начало коллекции.
    for offset in range(len(file_names)):
        for stratum in range(num_strata):
            key = (stratum, file_names[(stratum + offset) % len(file_names)])
            if key in buckets and key not in seen_keys:
                bucket_order.append(key)
                seen_keys.add(key)

    ordered_buckets = []
    for key in bucket_order:
        candidates = buckets[key]
        rng.shuffle(candidates)
        ordered_buckets.append(candidates)

    selected = []
    while ordered_buckets and len(selected) < num_questions:
        remaining_buckets = []
        for bucket in ordered_buckets:
            if bucket and len(selected) < num_questions:
                selected.append(bucket.pop())
            if bucket:
                remaining_buckets.append(bucket)
        ordered_buckets = remaining_buckets

    return selected


def generate_synthetic_dataset_from_db(
    config: RAGConfig, 
    collection_name: str,
    output_file: str, 
    num_questions: int = 20
) -> None:
    """
    Генерирует золотой датасет, выбирая случайные чанки из уже готовой коллекции Qdrant.
    """
    # --- 1. ИНИЦИАЛИЗАЦИЯ МИНИМАЛЬНОГО НАБОРА СЕРВИСОВ ---
    embedder = EmbeddingService(config.embedding)
    # QdrantService требует embedder для инициализации
    vector_db = QdrantService(config.vector_store, embedder=embedder)
    llm = OllamaLLMService(config.llm)

    # --- 2. ИЗВЛЕЧЕНИЕ СЛУЧАЙНЫХ ЧАНКОВ ИЗ QDRANT ---
    print(f"🔍 Подключаюсь к коллекции '{collection_name}'...")
    
    if not vector_db.collection_exists(collection_name):
        print(f"❌ Ошибка: Коллекция '{collection_name}' не найдена в Qdrant.")
        return

    print("📦 Выгружаю чанки для выбора случайных...")
    try:
        points = []
        offset = None
        while True:
            page, offset = vector_db.client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["text", "file_name", "chunk_index"],
                with_vectors=False,
            )
            points.extend(page)
            if offset is None:
                break
    except Exception as e:
        print(f"❌ Не удалось вычитать данные из Qdrant: {e}")
        return

    # Структурируем чанки: сохраняем словари, чтобы контролировать, что соседи из одного файла
    records_by_file = defaultdict(list)
    missing_chunk_index = 0
    for p in points:
        if p.payload and "text" in p.payload and "file_name" in p.payload:
            if "chunk_index" not in p.payload:
                missing_chunk_index += 1
                continue
            record = {
                "text": p.payload["text"],
                "file_name": p.payload["file_name"],
                "chunk_index": int(p.payload["chunk_index"]),
            }
            records_by_file[record["file_name"]].append(record)

    if missing_chunk_index:
        print(
            f"❌ У {missing_chunk_index} чанков нет chunk_index. "
            "Переиндексируйте коллекцию перед генерацией датасета."
        )
        return

    total_records = sum(len(records) for records in records_by_file.values())
    if not total_records:
        print("⚠️ В коллекции нет подходящих чанков.")
        return

    for file_name, records in records_by_file.items():
        index_map = {record["chunk_index"]: record for record in records}
        if len(index_map) != len(records):
            print(
                f"❌ В файле '{file_name}' есть дублирующиеся chunk_index. "
                "Переиндексируйте коллекцию."
            )
            return
    print(f"📊 Всего доступно чанков: {total_records}")
    target_records = select_stratified_targets(records_by_file, num_questions)
    actual_num_questions = len(target_records)
    
    golden_dataset: List[Dict[str, Any]] = []

    print(f"🚀 Запуск генерации датасета по {actual_num_questions} золотым чанкам...")

    for step, target_record in enumerate(target_records, 1):
        print(
            f"📝 Обработка {step}/{actual_num_questions} "
            f"({target_record['file_name']}, чанк {target_record['chunk_index']})..."
        )
        # Промпт даем строго по целевому чанку, чтобы вопрос был конкретным
        gen_prompt = f"""
        Based on the following text, generate one high-quality question and its detailed answer.
        The question must be answerable ONLY using the text provided. 
        Write both the question and the answer in Russian language.

        Format your response strictly as follows:
        QUESTION: [your question in Russian]
        ANSWER: [your answer in Russian]

        TEXT:
        {target_record["text"]}
        """

        try:
            response = llm.generate(gen_prompt)
            
            if "QUESTION:" in response and "ANSWER:" in response:
                question_part, answer_part = response.split("ANSWER:", maxsplit=1)
                question_part = question_part.removeprefix("QUESTION:").strip()
                answer_part = answer_part.strip()

                golden_dataset.append({
                    "question": question_part,
                    "contexts": [target_record["text"]],
                    "ground_truth": answer_part,
                    "source": {
                        "file_name": target_record["file_name"],
                        "chunk_index": target_record["chunk_index"],
                    },
                })
            else:
                print(f"⚠️ Модель нарушила формат. Пропускаю.")
                
        except Exception as e:
            print(f"⚠️ Ошибка на шаге {step}: {e}")

    # --- 4. СОХРАНЕНИЕ ---
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(golden_dataset, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Датасет успешно создан: {output_file}")
        print(f"📊 Всего примеров сгенерировано: {len(golden_dataset)}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении файла: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерация golden dataset с одним эталонным чанком"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--output", default="golden_dataset.json")
    parser.add_argument("--questions", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.error(f"Конфигурация не найдена: {args.config}")
    if args.questions < 1:
        parser.error("--questions должен быть больше нуля")

    rag_config = RAGConfig.from_yaml(args.config)
    generate_synthetic_dataset_from_db(
        config=rag_config,
        collection_name=rag_config.vector_store.collection_name,
        output_file=args.output,
        num_questions=args.questions,
    )

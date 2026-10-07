import json
import os
import random
from typing import List, Dict, Any

from src.config.schema import RAGConfig
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import QdrantService
from src.services.llm_service import OllamaLLMService

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
        # Загружаем пул точек (ограничимся 2000, чтобы не перегружать память, 
        # но иметь отличную выборку для рандома)
        points, _ = vector_db.client.scroll(
            collection_name=collection_name,
            limit=2000,
            with_payload=["text", "file_name"], # Забираем только текст и имя файла
            with_vectors=False                  # Векторы для генерации вопросов нам не нужны
        )
    except Exception as e:
        print(f"❌ Не удалось вычитать данные из Qdrant: {e}")
        return

    # Структурируем чанки: сохраняем словари, чтобы контролировать, что соседи из одного файла
    all_records = []
    for p in points:
        if p.payload and "text" in p.payload and "file_name" in p.payload:
            all_records.append({
                "text": p.payload["text"],
                "file_name": p.payload["file_name"]
            })
            
    total_records = len(all_records)
    if total_records < 3:
        print("⚠️ В базе слишком мало чанков для генерации расширенного контекста.")
        return

    print(f"📊 Всего доступно чанков: {total_records}")
    
    # Чтобы вопросы были разнообразными, выберем случайные индексы-мишени
    # Исключаем самый первый и самый последний чанк, чтобы у всех гарантированно были соседи
    possible_indices = list(range(1, total_records - 1))
    actual_num_questions = min(num_questions, len(possible_indices))
    target_indices = random.sample(possible_indices, actual_num_questions)
    
    golden_dataset: List[Dict[str, Any]] = []

    print(f"🚀 Запуск генерации датасета по {actual_num_questions} расширенным контекстам...")

    for step, idx in enumerate(target_indices, 1):
        print(f"📝 Обработка элемента {step}/{actual_num_questions} (Индекс чанка: {idx})...")
        
        target_record = all_records[idx]
        prev_record = all_records[idx - 1]
        next_record = all_records[idx + 1]
        
        # Собираем расширенное контекстное окно
        # Проверяем, что соседние чанки принадлежат тому же файлу (чтобы не смешивать разные документы)
        context_window = []
        
        if prev_record["file_name"] == target_record["file_name"]:
            context_window.append(prev_record["text"])
            
        context_window.append(target_record["text"]) # Сам "золотой" чанк-мишень
        
        if next_record["file_name"] == target_record["file_name"]:
            context_window.append(next_record["text"])

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
                parts = response.split("ANSWER:")
                question_part = parts[0].replace("QUESTION:", "").strip()
                answer_part = parts[1].strip()

                # КРИТИЧЕСКИЙ СДВИГ: В датасет сохраняем всё окно (3 чанка), а не один!
                golden_dataset.append({
                    "question": question_part,
                    "contexts": context_window,  # Теперь здесь список из 2-3 смежных чанков
                    "ground_truth": answer_part 
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
    CONFIG_PATH = "src/config/config.yaml"
    OUTPUT_JSON = "golden_dataset.json"
    MAX_QUESTIONS = 20 

    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Ошибка: Конфигурация не найдена: {CONFIG_PATH}")
    else:
        # Загружаем основной конфиг
        rag_config = RAGConfig.from_yaml(CONFIG_PATH)
        
        # Имя коллекции вытягиваем из конфига
        target_collection = rag_config.vector_store.collection_name
        
        # Запускаем процесс генерации напрямую из БД
        generate_synthetic_dataset_from_db(
            config=rag_config,
            collection_name=target_collection,
            output_file=OUTPUT_JSON,
            num_questions=MAX_QUESTIONS
        )

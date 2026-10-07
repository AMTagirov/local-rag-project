import json
import os
from typing import List, Dict, Any
from itertools import islice

from src.config.schema import RAGConfig
from src.services.document_parser import PDFDocumentParser
from src.services.text_splitter import ChunkTextSplitter
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import QdrantService
from src.services.llm_service import OllamaLLMService
from src.pipelines.rag_pipeline import RAGPipeline
from src.services.sparse_embedding_service import SparseEmbeddingService

def generate_synthetic_dataset(
    config: RAGConfig, 
    pdf_path: str, 
    output_file: str, 
    num_chunks: int = 20
) -> None:
    """
    Генерирует золотой датасет (вопрос, контекст, эталонный ответ) на основе документов.
    Полностью адаптирован под новую потоковую архитектуру и защищен от OOM.
    """
    
    # --- 1. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ (ИСПРАВЛЕНО: соблюдены зависимости) ---
    parser = PDFDocumentParser()
    embedder = EmbeddingService(config.embedding)
    
    # Передаем embedder в сплиттер и векторное хранилище
    splitter = ChunkTextSplitter(config.splitter, embedder=embedder)
    vector_db = QdrantService(config.vector_store, embedder=embedder)
    
    sparse_embedder = SparseEmbeddingService()
    llm = OllamaLLMService(config.llm)

    # Инициализируем пайплайн со всеми необходимыми зависимостями
    pipeline = RAGPipeline(
        config=config,
        parser=parser,
        splitter=splitter,
        embedder=embedder,
        vector_store=vector_db,
        llm=llm,
        sparse_service=sparse_embedder
    )

    # --- 2. ПОДГОТОВКА ПОТОКА ТЕКСТА (ИСПРАВЛЕНО: переход на стриминг) ---
    print(f"📖 Чтение файла в потоковом режиме: {pdf_path}")
    
    # Получаем ленивый генератор страниц
    pages_stream = parser.parse(pdf_path)
    
    # Получаем ленивый генератор семантических чанков
    chunks_stream = splitter.split_stream(pages_stream)
    
    # Безопасно откусываем ровно num_chunks чанков без загрузки всего файла в RAM
    chunks_to_process = list(islice(chunks_stream, num_chunks))
    
    golden_dataset: List[Dict[str, Any]] = []
    total_to_process = len(chunks_to_process)

    print(f"🚀 Начинаем генерацию датасета из {total_to_process} чанков...")

    # --- 3. ЦИКЛ ГЕНЕРАЦИИ ---
    for i, chunk in enumerate(chunks_to_process):
        print(f"📝 Обработка чанка {i+1}/{total_to_process}...")

        gen_prompt = f"""
        Based on the following text, generate one high-quality question and its detailed answer.
        The question must be answerable ONLY using the text provided. 
        Write both the question and the answer in Russian language.

        Format your response strictly as follows:
        QUESTION: [your question in Russian]
        ANSWER: [your answer in Russian]

        TEXT:
        {chunk}
        """

        try:
            response = llm.generate(gen_prompt)
            
            if "QUESTION:" in response and "ANSWER:" in response:
                parts = response.split("ANSWER:")
                question_part = parts[0].replace("QUESTION:", "").strip()
                answer_part = parts[1].strip()

                golden_dataset.append({
                    "question": question_part,
                    "contexts": [chunk],  
                    "ground_truth": answer_part 
                })
            else:
                print(f"⚠️ Предупреждение: Модель не соблюла формат на чанке {i+1}. Ответ: {response[:50]}...")
                
        except Exception as e:
            print(f"⚠️ Ошибка на чанке {i+1}: {e}")

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
    PDF_FILE = "Document.pdf"
    OUTPUT_JSON = "golden_dataset.json"
    MAX_QUESTIONS = 20 

    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Ошибка: Конфигурация не найдена: {CONFIG_PATH}")
    elif not os.path.exists(PDF_FILE):
        print(f"❌ Ошибка: Файл {PDF_FILE} не найден.")
    else:
        rag_config = RAGConfig.from_yaml(CONFIG_PATH)
        
        generate_synthetic_dataset(
            config=rag_config,
            pdf_path=PDF_FILE,
            output_file=OUTPUT_JSON,
            num_chunks=MAX_QUESTIONS
        )
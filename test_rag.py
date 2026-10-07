import os
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama

# --- КОНФИГУРАЦИЯ ---
PDF_PATH = "Document.pdf"  # Твой файл
COLLECTION_NAME = "my_first_rag"
MODEL_NAME = "llama3"  # Убедись, что скачал: ollama pull llama3
EMBED_MODEL = "models/all-MiniLM-L6-v2/" # Легкая модель для векторов
OLLAMA_HOST = "http://172.22.48.1:11434" # ЗАМЕНИТЕ НА ВАШ IP

def run_mvp():
    # 1. Инициализация моделей
    print("--- Инициализация моделей ---")
    try:
        embed_model = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        print(f"Ошибка инициализации эмбеддингов: {e}")
        return

    try:
        qdrant = QdrantClient(host="localhost", port=6333)
        # Проверка связи с Qdrant
        qdrant.get_collections()
    except Exception as e:
        print(f"Ошибка подключения к Qdrant (проверь Docker!): {e}")
        return

    # 2. Подготовка Qdrant (создание коллекции)
    
    #Динамически узнаем размерность эмбеддинга
    embedding_dimension = embed_model.get_sentence_embedding_dimension()
    
    if not qdrant.collection_exists(COLLECTION_NAME):
        print(f"Создание коллекции {COLLECTION_NAME}...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dimension,
                distance=Distance.COSINE),
        )
    else:
        print(f"Коллекция {COLLECTION_NAME} уже существует.")

    # 3. Парсинг PDF и разбиение на чанки
    print("--- Чтение и разбиение документа ---")
    if not os.path.exists(PDF_PATH):
        print(f"ОШИБКА: Файл {PDF_PATH} не найден!")
        return

    try:
        doc = pymupdf.open(PDF_PATH)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(full_text)
        print(f"Создано чанков: {len(chunks)}")
    except Exception as e:
        print(f"Ошибка при парсинге PDF: {e}")
        return

    # 4. Векторизация и сохранение в Qdrant
    print("--- Векторизация и сохранение в базу ---")
    points = []
    for i, chunk in enumerate(chunks):
        vector = embed_model.encode(chunk).tolist()
        points.append(PointStruct(
            id=i,
            vector=vector,
            payload={"text": chunk}
        ))
    
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Загружено {len(points)} точек в Qdrant.")

    # 5. Поиск (Retrieval)
    query = input("\nВведите ваш вопрос к документу: ")
    if not query:
        print("Вопрос не может быть пустым.")
        return
        
    print(f"Ищу ответ на вопрос: {query}")
    query_vector = embed_model.encode(query).tolist()
    
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        with_payload=True,
        limit=3
    ).points

    context = "\n".join([res.payload["text"] for res in search_result])

    # 6. Генерация (Generation)
    print("--- Генерация ответа через Ollama ---")
    prompt = f"""Используй только предоставленный контекст, чтобы ответить на вопрос.
Если в контексте нет ответа, скажи, что не знаешь. Отвечай на русском языке.

Контекст:
{context}

Вопрос: {query}

Ответ:"""

    try:
        # Используем Client для подключения к удаленному хосту (Windows)
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt},
        ])
        print("\n=== ОТВЕТ LLM ===")
        print(response['message']['content'])
    except Exception as e:
        print(f"Ошибка при обращении к Ollama: {e}")

if __name__ == "__main__":
    run_mvp()
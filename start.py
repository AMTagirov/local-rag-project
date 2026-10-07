import subprocess
import time
import sys
import requests
import os
import traceback
import shutil
import webbrowser

# Импорт компонентов вашей RAG-архитектуры для синхронизации базы
from src.config.schema import RAGConfig
from src.services.document_parser import PDFDocumentParser
from src.services.text_splitter import ChunkTextSplitter
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import QdrantService
from src.services.sparse_embedding_service import SparseEmbeddingService
from src.services.llm_service import OllamaLLMService
from src.services.reranker_service import RerankerService
from src.pipelines.rag_pipeline import RAGPipeline

# --- НАСТРОЙКИ И ПУТИ ---
CONFIG_PATH = "src/config/config.yaml"
QDRANT_URL = "http://127.0.0.1:6333"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen2.5:14b"
APP_SCRIPT = "app.py"         # Скрипт интерфейса Chainlit
CHAINLIT_PORT = "8000"         # Стандартный порт Chainlit

def wait_for_service(url, name, timeout=30):
    """Ждет, пока сервис станет доступен по HTTP."""
    print(f"⏳ Ожидание доступности {name} ({url})...", end="", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if "6333" in url:
                response = requests.get(url, timeout=2)  # Проверка Qdrant
            else:
                response = requests.get(f"{url}/api/tags", timeout=2)  # Проверка Ollama
                
            if response.status_code == 200:
                print(" ✅ Готов!")
                return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    print(f" ❌ Ошибка: {name} не ответил за {timeout}с")
    return False

def ensure_ollama():
    """Запускает сервер Ollama в фоне, если он не работает."""
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        print(" ✅ Ollama уже запущена.")
        return None
    except Exception:
        print("🚀 Ollama не обнаружена в фоне. Запускаю 'ollama serve'...")
        try:
            proc = subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            time.sleep(1)
            return proc
        except Exception as e:
            print(f" ❌ Не удалось запустить Ollama через subprocess: {e}")
            return None

def pull_ollama_model(model_name):
    """Гарантирует, что необходимая модель скачана в Ollama."""
    print(f"📦 Проверка наличия модели '{model_name}' в Ollama...")
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            print(f" ❌ Ollama вернула ошибку ответа: {response.status_code}")
            return False
            
        models_data = response.json().get('models', [])
        models = [m.get('name', '') for m in models_data if isinstance(m, dict)]
        
        target = model_name if ":" in model_name else f"{model_name}:latest"
        
        if any(target in m or m in target for m in models if m):
            print(f" ✅ Модель '{model_name}' уже установлена.")
            return True
        else:
            print(f" 📥 Модель '{model_name}' не найдена. Начинаю скачивание (ollama pull)...")
            print("⚠️ Это займет время (размер модели ~9 ГБ)...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f" ✅ Модель '{model_name}' успешно скачана.")
            return True
    except Exception as e:
        print(f" ❌ Ошибка при проверке/скачивании модели: {e}")
        return False

def run_vector_db_synchronization():
    """Инициализирует RAG-пайплайн и запускает синхронизацию локальной папки с Qdrant."""
    print("📂 Шаг 3.5: Запуск синхронизации локальной папки с Qdrant...")
    if not os.path.exists(CONFIG_PATH):
        print(f" ❌ Ошибка синхронизации: Конфиг не найден по пути '{CONFIG_PATH}'")
        return False

    try:
        # 1. Загрузка конфигурации
        config = RAGConfig.from_yaml(CONFIG_PATH)
        docs_dir = config.vector_store.docs_dir

        if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
            print(f" 📂 Папка документов '{docs_dir}' пуста или не существует. Пропускаем векторизацию.")
            return True

        print(f" 📖 Обнаружены локальные файлы в '{docs_dir}'. Сборка пайплайна для индексации...")
        
        # 2. Инициализация необходимых сервисов
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

        # 4. Вызов синхронизации директории (синхронно, так как мы в start.py)
        print(" ⏳ Выполняется индексация новых документов (парсинг, чанкинг, эмбеддинги)...")
        pipeline.sync_directory()
        print(" ✅ Синхронизация векторной базы успешно завершена!")
        return True

    except Exception as e:
        print(f" ❌ Критическая ошибка при синхронизации базы данных: {e}")
        traceback.print_exc()
        return False

def main():
    background_processes = []
    docker_started = False

    try:
        # ПРОВЕРКА СЛУЖБЫ DOCKER
        docker_check = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if docker_check.returncode != 0:
            print("🐳 Демон Docker не активен. Пробуем запустить службу...")
            subprocess.run(["sudo", "service", "docker", "start"])
            time.sleep(2)

        # 1. ЗАПУСК QDRANT
        print("🚀 Шаг 1: Запуск инфраструктуры (Qdrant via Docker)...")
        result = subprocess.run(["docker", "compose", "up", "-d"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f" ❌ Ошибка docker compose: {result.stderr.decode()}")
            return
        
        docker_started = True
        if not wait_for_service(QDRANT_URL, "Qdrant"): 
            return

        # 2. ЗАПУСК OLLAMA SERVER
        print("🚀 Шаг 2: Запуск сервера Ollama...")
        ollama_proc = ensure_ollama()
        if ollama_proc:
            background_processes.append(ollama_proc)
        
        if not wait_for_service(OLLAMA_URL, "Ollama"): 
            return

        # 3. ПРОВЕРКА И ПОДГОТОВКА МОДЕЛИ
        print("🚀 Шаг 3: Подготовка LLM-модели...")
        if not pull_ollama_model(MODEL_NAME):
            print("❌ Не удалось подготовить модель. Прерывание.")
            return

        # --- СИНХРОНИЗАЦИЯ БАЗЫ ЗНАНИЙ (НОВЫЙ БЛОК) ---
        if not run_vector_db_synchronization():
            print("⚠️ Предупреждение: Скрипт продолжит запуск UI, но новые файлы могли не проиндексироваться.")

        # 4. ЗАПУСК ВЕБ-ИНТЕРФЕЙСА CHAINLIT
        print(f"🚀 Шаг 4: Запуск веб-интерфейса через Chainlit: {APP_SCRIPT}")
        if not os.path.exists(APP_SCRIPT):
            print(f" ❌ Ошибка: Файл интерфейса '{APP_SCRIPT}' не найден в текущей папке!")
            return
            
        chainlit_path = shutil.which("chainlit")
        if not chainlit_path:
            print(" ❌ Ошибка: Утилита 'chainlit' не найдена. Вы активировали venv?")
            return

        # Запуск сервера: chainlit run app.py --port 8000
        app_proc = subprocess.Popen([chainlit_path, "run", APP_SCRIPT, "--port", CHAINLIT_PORT])
        background_processes.append(app_proc)

        # 5. АВТОМАТИЧЕСКИЙ СТАРТ БРАУЗЕРА
        print("🌐 Инициализация веб-интерфейса...")
        time.sleep(3)  # Пауза, чтобы Chainlit успел поднять сокеты

        url = f"http://localhost:{CHAINLIT_PORT}"

        # Надежный кроссплатформенный запуск браузера (с учетом WSL2)
        if "WSL_DISTRO_NAME" in os.environ:
            os.system(f"cmd.exe /c start {url}")
        elif sys.platform == "win32":
            os.system(f"start {url}")
        elif sys.platform == "darwin":
            os.system(f"open {url}")
        else:
            try:
                webbrowser.open(url)
            except Exception:
                print(f"🔗 Не удалось открыть браузер автоматически. Перейдите вручную: {url}")

        # Удерживаем скрипт активным, пока работает UI
        app_proc.wait()

    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал остановки (Ctrl+C)...")
    except Exception as e:
        print("\n💥 Критическая ошибка во время работы скрипта:")
        traceback.print_exc()
    finally:
        print("\n🧹 Очистка временных ресурсов и завершение процессов...")
        for p in background_processes:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        
        if docker_started:
            print("🐳 Остановка Docker-контейнеров (Qdrant)...")
            subprocess.run(["docker", "compose", "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        print("✅ Все службы успешно остановлены. Оперативная память очищена.")

if __name__ == "__main__":
    main()

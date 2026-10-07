import requests
from src.core.interfaces import LLMService

class LlamaCppLLMService(LLMService):
    def __init__(self, base_url: str, model_name: str = "model"):
        """
        Инициализирует сервис для работы с llama.cpp server.
        :param base_url: URL сервера (например, http://localhost:8080/v1)
        :param model_name: Имя модели (в llama.cpp часто можно оставить любое)
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ через OpenAI-совместимый API llama.cpp.
        """
        try:
            # Используем эндпоинт completions (OpenAI compatible)
            url = f"{self.base_url}/completions"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 1024,
                "stop": ["<|endoftext|>", "</s>"] # Добавьте стоп-токены вашей модели
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['text'].strip()
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при запросе к llama.cpp: {e}")
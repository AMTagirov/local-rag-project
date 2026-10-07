from typing import Generator
import ollama
from src.core.interfaces import LLMService
from src.config.schema import LLMConfig

class OllamaLLMService(LLMService):
    def __init__(self, config: LLMConfig):
        """
        Инициализирует сервис на основе конфигурации.
        """
        self.config = config
        # Используем Client для возможности указать base_url из конфига
        self.client = ollama.Client(host=config.base_url)

    def generate(self, prompt: str) -> str:
        """
        Синхронная генерация ответа (используется в скриптах оценки и датасетов).
        Полностью детерминирована на основе параметров из конфига.
        """
        try:
            response = self.client.generate(
                model=self.config.model_name,
                prompt=prompt,
                options={
                    "num_ctx": self.config.num_ctx,
                    "temperature": getattr(self.config, "temperature", 0.0),
                    "top_p": getattr(self.config, "top_p", 1.0),
                    "seed": getattr(self.config, "seed", 42)
                }
            )
            return response['response']
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации ответа через Ollama: {e}")

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Потоковая генерация ответа (будет использоваться в веб-интерфейсе app.py).
        Возвращает генератор токенов по мере их создания моделью Qwen.
        """
        try:
            # Вызываем generate с флагом stream=True в библиотеке ollama
            stream = self.client.generate(
                model=self.config.model_name,
                prompt=prompt,
                stream=True,  # Включаем потоковый режим на стороне Ollama
                options={
                    "num_ctx": self.config.num_ctx,
                    "temperature": getattr(self.config, "temperature", 0.0),
                    "top_p": getattr(self.config, "top_p", 1.0),
                    "seed": getattr(self.config, "seed", 42)
                }
            )
            for chunk in stream:
                yield chunk['response']
                
        except Exception as e:
            raise RuntimeError(f"Ошибка при потоковой генерации через Ollama: {e}")
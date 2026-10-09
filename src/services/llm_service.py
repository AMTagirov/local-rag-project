from typing import AsyncGenerator, Optional
import ollama
from src.core.interfaces import LLMService
from src.config.schema import LLMConfig

class OllamaLLMService(LLMService):
    def __init__(self, config: LLMConfig):
        """
        Инициализирует сервис на основе конфигурации.
        """
        self.config = config
        # Синхронный клиент используется в CLI и скриптах оценки.
        self.client = ollama.Client(host=config.base_url)
        # Асинхронный клиент не блокирует event loop Chainlit между токенами.
        self.async_client = ollama.AsyncClient(host=config.base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Синхронная генерация ответа (используется в скриптах оценки и датасетов).
        Полностью детерминирована на основе параметров из конфига.
        """
        try:
            response = self.client.generate(
                model=self.config.model_name,
                prompt=prompt,
                system=system_prompt or "",
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

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Потоковая генерация ответа (будет использоваться в веб-интерфейсе app.py).
        Возвращает генератор токенов по мере их создания моделью Qwen.
        """
        try:
            stream = await self.async_client.generate(
                model=self.config.model_name,
                prompt=prompt,
                system=system_prompt or "",
                stream=True,
                options={
                    "num_ctx": self.config.num_ctx,
                    "temperature": getattr(self.config, "temperature", 0.0),
                    "top_p": getattr(self.config, "top_p", 1.0),
                    "seed": getattr(self.config, "seed", 42)
                }
            )
            async for chunk in stream:
                yield chunk['response']
                
        except Exception as e:
            raise RuntimeError(f"Ошибка при потоковой генерации через Ollama: {e}")

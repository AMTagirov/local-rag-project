from typing import List, Tuple
from sentence_transformers import CrossEncoder
from src.config.schema import RerankerConfig

class RerankerService:
    """
    Сервис для переранжирования результатов поиска.
    Использует Cross-Encoder модели для оценки релевантности пары (вопрос, текст).
    """

    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = CrossEncoder(config.model_name)

    def rerank(self, query: str, passages: List[str]) -> List[int]:
        """
        Принимает запрос и список текстов, возвращает индексы текстов, 
        отсортированные по убыванию релевантности.

        :param query: Текст вопроса.
        :param passages: Список текстов (контекстов), полученных из векторного поиска.
        :return: Список индексов (int), отсортированных по качеству.
        """
        if not passages:
            return []

        # Подготовка пар (запрос, текст) для Cross-Encoder
        # Модель ожидает список пар: [[query, p1], [query, p2], ...]
        pairs = [[query, passage] for passage in passages]
        
        # Получаем скоры (релевантность)
        scores = self.model.predict(pairs)

        # Сортируем индексы по убыванию скоров
        # argsort: получаем индексы, которые бы отсортировали массив по возрастанию, 
        # поэтому используем [::-1] для убывания.
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        
        return sorted_indices
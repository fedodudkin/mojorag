"""
Векторизация текста через sentence-transformers.
Поддержка ONNX для оптимальной производительности.
"""

import numpy as np
from pathlib import Path
from typing import List, Union
import logging

try:
    from .config import get_profile, EMBED_MODEL_NAME, EMBED_BACKEND, EMBED_DIMENSION
except ImportError:
    # Fallback for running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_profile, EMBED_MODEL_NAME, EMBED_BACKEND, EMBED_DIMENSION

logger = logging.getLogger(__name__)


class Embedder:
    """Класс для векторизации текста."""
    
    def __init__(self, model_name: str = None, backend: str = None):
        """
        Инициализация модели эмбеддингов.
        
        Args:
            model_name: Имя модели эмбеддингов
            backend: Бэкенд ('onnx', 'torch', 'transformers')
        """
        self.model_name = model_name or EMBED_MODEL_NAME
        self.backend = backend or EMBED_BACKEND
        self.profile = get_profile()
        
        # Загрузка модели
        self.model = self._load_model()
        self.dimension = EMBED_DIMENSION
    
    def _load_model(self):
        """Загружает модель эмбеддингов."""
        try:
            from sentence_transformers import SentenceTransformer
            from .config import MODELS_DIR
            
            logger.info(f"Загрузка модели {self.model_name} с бэкендом {self.backend}")
            
            # Используем локальную модель если она доступна
            local_model_path = MODELS_DIR / self.model_name
            if local_model_path.exists():
                logger.info(f"Используем локальную модель: {local_model_path}")
                model_path = str(local_model_path)
            else:
                logger.info(f"Локальная модель не найдена, используем HuggingFace: {self.model_name}")
                model_path = self.model_name
            
            # Конфигурация для оптимальной производительности
            model_kwargs = {}
            config_kwargs = {}
            
            if self.backend == "onnx":
                # ONNX для лучшей производительности
                model_kwargs = {
                    "file_name": "model.onnx",
                    "provider": "CPUExecutionProvider"
                }
            elif self.backend == "torch":
                # Torch с оптимизациями
                config_kwargs = {
                    "use_flash_attn": False,  # Отключаем flash attention для стабильности
                }
            
            model = SentenceTransformer(
                model_path,
                backend=self.backend,
                model_kwargs=model_kwargs,
                config_kwargs=config_kwargs
            )
            
            # Оптимизация для инференса
            if hasattr(model, 'encode'):
                model.eval()  # Режим инференса
            
            logger.info(f"Модель {self.model_name} успешно загружена")
            return model
            
        except ImportError:
            raise ImportError(
                "sentence-transformers не установлен. "
                "Установите: pip install sentence-transformers[onnx]"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {self.model_name}: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = None) -> np.ndarray:
        """
        Векторизует текст(ы).
        
        Args:
            texts: Текст или список текстов для векторизации
            batch_size: Размер батча (если None, используется из профиля)
            
        Returns:
            numpy массив с эмбеддингами
        """
        if batch_size is None:
            batch_size = self.profile.embed_batch_size
        
        # Нормализация входа
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValueError("texts должен быть строкой или списком строк")
        
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        
        try:
            # Векторизация с оптимальными параметрами
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,  # Нормализация для cosine similarity
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Ошибка векторизации: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Векторизует один текст.
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Вектор эмбеддингов
        """
        return self.encode(text)[0]
    
    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """
        Вычисляет косинусное сходство между запросом и документами.
        
        Args:
            query_embedding: Эмбеддинг запроса
            document_embeddings: Эмбеддинги документов
            
        Returns:
            Массив сходств
        """
        # Косинусное сходство (эмбеддинги уже нормализованы)
        return np.dot(document_embeddings, query_embedding)
    
    def get_embedding_info(self) -> dict:
        """Возвращает информацию о модели эмбеддингов."""
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "dimension": self.dimension,
            "batch_size": self.profile.embed_batch_size,
            "max_seq_length": getattr(self.model, 'max_seq_length', None),
        }
    
    def benchmark(self, sample_texts: List[str] = None) -> dict:
        """
        Бенчмарк производительности векторизации.
        
        Args:
            sample_texts: Тексты для теста
            
        Returns:
            Результаты бенчмарка
        """
        import time
        
        if sample_texts is None:
            # Генерируем тестовые тексты
            sample_texts = [
                "Это тестовый текст для бенчмарка производительности эмбеддингов.",
                "MojoRAG - высокопроизводительная система для поиска по документам.",
                "Векторизация текста позволяет семантический поиск по смыслу.",
                "Sentence-transformers предоставляет качественные эмбеддинги.",
                "ONNX бэкенд оптимизирует вычисления на CPU.",
            ] * 20  # 100 текстов
        
        logger.info(f"Бенчмарк на {len(sample_texts)} текстах")
        
        # Тест производительности
        start_time = time.time()
        embeddings = self.encode(sample_texts)
        end_time = time.time()
        
        processing_time = end_time - start_time
        texts_per_second = len(sample_texts) / processing_time
        
        return {
            "total_texts": len(sample_texts),
            "processing_time": processing_time,
            "texts_per_second": texts_per_second,
            "avg_time_per_text": processing_time / len(sample_texts),
            "embedding_shape": embeddings.shape,
            "memory_usage_mb": embeddings.nbytes / (1024 * 1024),
        }


class CachedEmbedder(Embedder):
    """Эмбеддер с кэшированием для повторных запросов."""
    
    def __init__(self, model_name: str = None, backend: str = None, cache_size: int = 1000):
        """
        Инициализация с кэшем.
        
        Args:
            model_name: Имя модели
            backend: Бэкенд
            cache_size: Размер кэша
        """
        super().__init__(model_name, backend)
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = None) -> np.ndarray:
        """Векторизация с кэшированием."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Проверяем кэш
        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                embeddings.append(None)  # Заполнитель
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Векторизуем только новые тексты
        if texts_to_encode:
            new_embeddings = super().encode(texts_to_encode, batch_size)
            
            # Обновляем кэш
            for text, embedding in zip(texts_to_encode, new_embeddings):
                self._add_to_cache(text, embedding)
            
            # Вставляем новые эмбеддинги в результат
            for idx, embedding in zip(indices_to_encode, new_embeddings):
                embeddings[idx] = embedding
        
        return np.array(embeddings)
    
    def _add_to_cache(self, text: str, embedding: np.ndarray):
        """Добавляет эмбеддинг в кэш."""
        # Удаляем старые записи если кэш переполнен
        if len(self._cache) >= self.cache_size:
            oldest_text = self._cache_order.pop(0)
            del self._cache[oldest_text]
        
        self._cache[text] = embedding
        self._cache_order.append(text)
    
    def clear_cache(self):
        """Очищает кэш."""
        self._cache.clear()
        self._cache_order.clear()
    
    def get_cache_stats(self) -> dict:
        """Возвращает статистику кэша."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_hit_ratio": getattr(self, '_cache_hits', 0) / max(1, getattr(self, '_cache_requests', 1)),
        }

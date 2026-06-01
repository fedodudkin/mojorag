"""
Поиск по векторному индексу через Mojo-ядро.
Высокопроизводительный поиск с метаданными.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

try:
    from .config import get_vector_index_path, get_metadata_path
except ImportError:
    # Fallback для запуска как скрипта
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_vector_index_path, get_metadata_path

logger = logging.getLogger(__name__)

# Импорт Mojo через единый источник (__init__.py)
try:
    from . import mojorag_core, is_mojo_available
    _MOJO_AVAILABLE = is_mojo_available()
except (ImportError, AttributeError):
    import sys
    from pathlib import Path
    _MOJO_PATH = Path(__file__).resolve().parent.parent / "mojo"
    if _MOJO_PATH.exists():
        sys.path.insert(0, str(_MOJO_PATH))
    try:
        import mojorag_core
        _MOJO_AVAILABLE = True
    except ImportError:
        _MOJO_AVAILABLE = False
        import logging
        logging.warning("Mojo модуль недоступен, будет использоваться Python fallback")


class MojoRetriever:
    """Класс для поиска по векторному индексу."""
    
    def __init__(self):
        """Инициализация поисковика."""
        self.vectors = None
        self.metadata = []
        self._loaded = False
    
    def index_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Сохраняет векторы и метаданные в индекс.
        
        Args:
            vectors: Матрица эмбеддингов [N, D]
            metadata: Список метаданных для каждого вектора
        """
        if len(vectors) != len(metadata):
            raise ValueError("Количество векторов должно совпадать с количеством метаданных")
        
        if _MOJO_AVAILABLE:
            self._index_with_mojo(vectors, metadata)
        else:
            self._index_with_python(vectors, metadata)
        
        self.vectors = vectors
        self.metadata = metadata
        self._loaded = True
        
        logger.info(f"Проиндексировано {len(vectors)} векторов размерности {vectors.shape[1]}")
    
    def _index_with_mojo(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Сохранение индекса через Mojo."""
        try:
            # Конвертация numpy в Python list для Mojo
            vectors_list = vectors.tolist()
            
            # Сохранение векторов
            success = mojorag_core.save_index(vectors_list, str(get_vector_index_path()))
            if not success:
                raise RuntimeError("Ошибка сохранения индекса через Mojo")
            
            # Включить текст в метаданные ПЕРЕД сохранением
            combined_metadata = []
            for chunk in metadata:
                # chunk уже плоский dict из orchestrator.ingest()
                # Сохраняем как есть (уже содержит text + все метаданные)
                combined_metadata.append(dict(chunk))
            
            # Сохранение метаданных в JSON
            with open(get_metadata_path(), 'w', encoding='utf-8') as f:
                json.dump(combined_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info("Индекс успешно сохранен через Mojo")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения через Mojo: {e}")
            raise
    
    def _index_with_python(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Python fallback для сохранения индекса."""
        try:
            # Сохранение векторов в numpy формате
            np.save(get_vector_index_path().with_suffix('.npy'), vectors)
            
            # Включить текст в метаданные ПЕРЕД сохранением
            combined_metadata = []
            for chunk in metadata:
                # chunk уже плоский dict из orchestrator.ingest()
                # Сохраняем как есть (уже содержит text + все метаданные)
                combined_metadata.append(dict(chunk))
            
            # Сохранение метаданных в JSON
            with open(get_metadata_path(), 'w', encoding='utf-8') as f:
                json.dump(combined_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info("Индекс сохранен через Python fallback")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения через Python: {e}")
            raise
    
    def load(self) -> bool:
        """
        Загружает индекс из файлов.
        
        Returns:
            True если загрузка успешна
        """
        try:
            if _MOJO_AVAILABLE:
                return self._load_with_mojo()
            else:
                return self._load_with_python()
                
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}")
            return False
    
    def _load_with_mojo(self) -> bool:
        """Загрузка индекса через Mojo."""
        try:
            # Загрузка векторов
            vectors_list = mojorag_core.load_index(str(get_vector_index_path()))
            if vectors_list is None:
                return False
            
            self.vectors = np.array(vectors_list)
            
            # Загрузка метаданных
            with open(get_metadata_path(), 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self._loaded = True
            logger.info(f"Индекс загружен через Mojo: {len(self.vectors)} векторов")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки через Mojo: {e}")
            return False
    
    def _load_with_python(self) -> bool:
        """Python fallback для загрузки индекса."""
        try:
            # Загрузка векторов
            vectors_path = get_vector_index_path().with_suffix('.npy')
            if not vectors_path.exists():
                logger.error(f"Файл индекса не найден: {vectors_path}")
                return False
            
            self.vectors = np.load(vectors_path)
            
            # Загрузка метаданных
            with open(get_metadata_path(), 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self._loaded = True
            logger.info(f"Индекс загружен через Python: {len(self.vectors)} векторов")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки через Python: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск наиболее похожих векторов.
        
        Args:
            query_vector: Вектор запроса
            k: Количество результатов
            
        Returns:
            Список результатов с метаданными
        """
        if not self._loaded:
            raise RuntimeError("Индекс не загружен. Вызовите load() сначала")
        
        if _MOJO_AVAILABLE:
            return self._search_with_mojo(query_vector, k)
        else:
            return self._search_with_python(query_vector, k)
    
    def _search_with_mojo(self, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Поиск через Mojo ядро."""
        try:
            # Конвертация вектора запроса
            query_list = query_vector.tolist()
            vectors_list = self.vectors.tolist()
            
            # Поиск через Mojo
            results = mojorag_core.search_similar(query_list, vectors_list, k)
            
            # Обогащение результатами метаданными
            enriched_results = []
            for result in results:
                idx = result["index"]
                score = result["score"]
                
                if 0 <= idx < len(self.metadata):
                    enriched_result = self.metadata[idx].copy()
                    enriched_result["score"] = float(score)
                    enriched_result["rank"] = len(enriched_results) + 1
                    enriched_results.append(enriched_result)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска через Mojo: {e}")
            # Fallback на Python
            return self._search_with_python(query_vector, k)
    
    def _search_with_python(self, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Python fallback для поиска."""
        try:
            # Вычисление косинусного сходства
            similarities = np.dot(self.vectors, query_vector)
            
            # Получение top-k индексов
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            # Формирование результатов
            results = []
            for rank, idx in enumerate(top_k_indices):
                if 0 <= idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result["score"] = float(similarities[idx])
                    result["rank"] = rank + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска через Python: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику индекса."""
        if not self._loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "num_vectors": len(self.vectors),
            "dimension": self.vectors.shape[1],
            "num_metadata": len(self.metadata),
            "mojo_available": _MOJO_AVAILABLE,
            "using_mojo": _MOJO_AVAILABLE and self._loaded,
        }
    
    def is_loaded(self) -> bool:
        """Проверяет загружен ли индекс."""
        return self._loaded
    
    def clear(self) -> None:
        """Очищает загруженные данные."""
        self.vectors = None
        self.metadata = []
        self._loaded = False


class HybridRetriever(MojoRetriever):
    """Гибридный поисковик с дополнительными фильтрами."""
    
    def search(self, query_vector: np.ndarray, k: int = 5, 
               source_filter: str = None, 
               min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Расширенный поиск с фильтрацией.
        
        Args:
            query_vector: Вектор запроса
            k: Количество результатов
            source_filter: Фильтр по источнику
            min_score: Минимальный порог сходства
            
        Returns:
            Отфильтрованные результаты
        """
        # Базовый поиск
        results = super().search(query_vector, k * 2)  # Берем больше для фильтрации
        
        # Применение фильтров
        filtered_results = []
        for result in results:
            score = result["score"]
            
            # Фильтр по минимальному сходству
            if score < min_score:
                continue
            
            # Фильтр по источнику
            if source_filter:
                result_source = result.get("metadata", {}).get("source", "")
                if source_filter not in result_source:
                    continue
            
            result["rank"] = len(filtered_results) + 1
            filtered_results.append(result)
            
            # Ограничение по количеству
            if len(filtered_results) >= k:
                break
        
        return filtered_results

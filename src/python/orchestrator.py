"""
Оркестрация полного цикла RAG: вопрос → поиск → ответ.
Центральный координатор всех компонентов MojoRAG.
"""

from pathlib import Path
from typing import Generator, Iterator, List, Dict, Any, Union
import logging

try:
    from .config import get_profile, validate_environment, get_vector_index_path
    from .ingester import MarkdownIngester, DocumentChunk
    from .embedder import Embedder
    from .retriever import MojoRetriever
    from .generator import LLMGenerator
except ImportError:
    # Fallback для запуска как скрипта
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_profile, validate_environment, get_vector_index_path
    from ingester import MarkdownIngester, DocumentChunk
    from embedder import Embedder
    from retriever import MojoRetriever
    from generator import LLMGenerator

logger = logging.getLogger(__name__)


class MojoRAG:
    """Основной класс MojoRAG системы."""
    
    def __init__(self, profile_name: str = None):
        """
        Инициализация всех компонентов.
        
        Args:
            profile_name: Имя профиля или None для автоопределения
        """
        # Проверка окружения
        warnings = validate_environment()
        if warnings:
            logger.warning("Предупреждения окружения:")
            for warning in warnings:
                logger.warning(f"  {warning}")
        
        # Профиль
        self.profile = get_profile()
        
        # Компоненты
        self.ingester = MarkdownIngester()
        self.embedder = Embedder()
        self.retriever = MojoRetriever()
        self.generator = LLMGenerator()
        
        # Загрузка индекса если существует
        self._index_loaded = False
        self._try_load_index()
        
        logger.info(f"MojoRAG инициализирован с профилем: {self.profile.name}")
    
    def _try_load_index(self) -> bool:
        """Пытается загрузить существующий индекс."""
        try:
            if get_vector_index_path().exists():
                success = self.retriever.load()
                self._index_loaded = success
                if success:
                    logger.info("Индекс успешно загружен")
                return success
        except Exception as e:
            logger.warning(f"Ошибка загрузки индекса: {e}")
        
        self._index_loaded = False
        return False
    
    def ingest(self, data_dir: str) -> Dict[str, Any]:
        """
        Полный цикл индексации директории.
        
        Args:
            data_dir: Путь к директории с документами
            
        Returns:
            Статистика индексации
        """
        logger.info(f"Начало индексации директории: {data_dir}")
        
        # 1. Парсинг документов
        chunks = self.ingester.parse_directory(data_dir)
        if not chunks:
            raise ValueError("Не найдено документов для индексации")
        
        # 2. Векторизация
        logger.info("Векторизация документов...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        
        # 3. Сохранение индекса
        logger.info("Сохранение индекса...")
        # Сохраняем текст ВМЕСТЕ с метаданными
        metadata = []
        for chunk in chunks:
            meta = dict(chunk.metadata) if hasattr(chunk, 'metadata') else dict(chunk.get("metadata", {}))
            meta["text"] = chunk.text if hasattr(chunk, 'text') else chunk.get("text", "")
            metadata.append(meta)
        self.retriever.index_vectors(embeddings, metadata)
        
        # 4. Загрузка индекса в память
        self.retriever.load()
        self._index_loaded = True
        
        # Статистика
        stats = self.ingester.get_stats(chunks)
        stats.update({
            "profile": self.profile.name,
            "embedding_dimension": embeddings.shape[1],
            "index_size_mb": embeddings.nbytes / (1024 * 1024),
        })
        
        logger.info(f"Индексация завершена: {stats['total_chunks']} чанков")
        return stats
    
    def ask(
        self,
        question: str,
        k: int = 5,
        stream: bool = True,
        strict: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Полный цикл вопрос-ответ.
        
        Args:
            question: Вопрос пользователя
            k: Количество релевантных чанков
            stream: Стриминг ответа (True) или строка (False)
            strict: Строгий режим (только из контекста)
            
        Returns:
            Генератор токенов (stream=True) или строка (stream=False)
        """
        if not self._index_loaded:
            raise RuntimeError("Индекс не загружен. Сначала выполните ingest() или убедитесь что индекс существует")
        
        # 1. Векторизация вопроса
        query_embedding = self.embedder.encode_single(question)
        
        # 2. Поиск релевантных чанков
        results = self.retriever.search(query_embedding, k=k)
        
        if not results:
            msg = "К сожалению, я не нашел релевантной информации в базе знаний."
            if stream:
                return self._stream_message(msg)
            return msg
        
        # 3. Подготовка контекста
        context_chunks = []
        for result in results:
            meta = {k: v for k, v in result.items() if k not in ("text", "score", "rank")}
            context_chunks.append({
                "text": result.get("text", ""),
                "metadata": meta,
                "score": result.get("score", 0.0),
            })
        
        # 4. Генерация ответа
        gen_result = self.generator.generate(
            question=question,
            chunks=context_chunks,
            strict=strict,
            stream=stream,
            max_tokens=self.profile.llm_ctx // 2,
            temperature=0.7,
        )
        
        if stream:
            return self._stream_response(gen_result)
        return gen_result

    @staticmethod
    def _stream_message(msg: str) -> Generator[str, None, None]:
        yield msg

    @staticmethod
    def _stream_response(gen_result) -> Generator[str, None, None]:
        yield from gen_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает полную статистику системы."""
        stats = {
            "profile": {
                "name": self.profile.name,
                "description": self.profile.description,
                "llm_model": self.profile.llm_model,
                "llm_ctx": self.profile.llm_ctx,
                "llm_threads": self.profile.llm_threads,
                "embed_batch_size": self.profile.embed_batch_size,
            },
            "index_loaded": self._index_loaded,
        }
        
        if self._index_loaded:
            stats.update(self.retriever.get_stats())
        
        # Информация о моделях
        stats["models"] = {
            "embedder": self.embedder.get_embedding_info(),
        }
        
        return stats
    
    def reload_index(self) -> bool:
        """Перезагружает индекс из файлов."""
        return self._try_load_index()
    
    def unload_models(self):
        """Выгружает модели из памяти для освобождения ресурсов."""
        self.generator.unload()
        logger.info("Модели выгружены из памяти")
    
    def __del__(self):
        """Очистка ресурсов."""
        try:
            self.unload_models()
        except:
            pass


def create_mojorag(profile_name: str = None) -> MojoRAG:
    """
    Фабричная функция для создания MojoRAG.
    
    Args:
        profile_name: Имя профиля или None для автоопределения
        
    Returns:
        Экземпляр MojoRAG
    """
    return MojoRAG(profile_name)


# Удобные функции для быстрого использования
def quick_ask(question: str, data_dir: str = None, k: int = 5) -> str:
    """
    Быстрый вопрос-ответ без необходимости управлять объектами.
    
    Args:
        question: Вопрос
        data_dir: Директория с документами (если нужно переиндексировать)
        k: Количество релевантных чанков
        
    Returns:
        Ответ на вопрос
    """
    rag = MojoRAG()
    
    # Индексация если нужно
    if data_dir and not rag._index_loaded:
        rag.ingest(data_dir)
    
    return rag.ask(question, k=k, stream=False)

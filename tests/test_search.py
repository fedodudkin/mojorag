"""
Тесты для поиска и векторизации.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.python.embedder import Embedder
from src.python.retriever import MojoRetriever


class TestEmbedder:
    """Тесты для эмбеддера."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.embedder = Embedder()
    
    def test_single_text_embedding(self):
        """Тест эмбеддинга одного текста."""
        text = "Это тестовый текст для эмбеддинга"
        embedding = self.embedder.encode_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # Размерность all-MiniLM-L6-v2
        assert np.allclose(np.linalg.norm(embedding), 1.0)  # Нормализация
    
    def test_batch_embedding(self):
        """Тест пакетного эмбеддинга."""
        texts = [
            "Первый текст",
            "Второй текст", 
            "Третий текст"
        ]
        embeddings = self.embedder.encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)  # Нормализация
    
    def test_empty_text(self):
        """Тест пустого текста."""
        embedding = self.embedder.encode_single("")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
    
    def test_similarity_computation(self):
        """Тест вычисления сходства."""
        text1 = "Поиск информации"
        text2 = "Информационный поиск"
        text3 = "Кулинарный рецепт"
        
        emb1 = self.embedder.encode_single(text1)
        emb2 = self.embedder.encode_single(text2)
        emb3 = self.embedder.encode_single(text3)
        
        # Похожие тексты должны иметь высокое сходство
        sim12 = self.embedder.compute_similarity(emb1, emb2)
        sim13 = self.embedder.compute_similarity(emb1, emb3)
        
        assert sim12 > sim13
        assert sim12 > 0.7  # Высокое сходство для похожих текстов


class TestMojoRetriever:
    """Тесты для Mojo поисковика."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.retriever = MojoRetriever()
        
        # Создаем тестовые данные
        self.vectors = np.random.rand(10, 384).astype(np.float32)
        self.metadata = [
            {"text": f"Текст {i}", "source": f"file_{i}.md"}
            for i in range(10)
        ]
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_index_and_search(self):
        """Тест индексации и поиска."""
        # Индексация
        self.retriever.index_vectors(self.vectors, self.metadata)
        
        # Поиск
        query_vector = self.vectors[0]  # Используем первый вектор как запрос
        results = self.retriever.search(query_vector, k=3)
        
        assert len(results) <= 3
        assert all("score" in result for result in results)
        assert all("source" in result for result in results)
        assert all("text" in result for result in results)
        
        # Первый результат должен иметь самый высокий score
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_with_k_greater_than_available(self):
        """Тест поиска с k больше количества векторов."""
        self.retriever.index_vectors(self.vectors, self.metadata)
        
        query_vector = self.vectors[0]
        results = self.retriever.search(query_vector, k=20)  # Больше чем 10
        
        assert len(results) == 10  # Должны вернуть все доступные
    
    def test_empty_index_search(self):
        """Тест поиска по пустому индексу."""
        with pytest.raises(RuntimeError):
            self.retriever.search(np.random.rand(384), k=5)


@pytest.mark.skipif(
    not Path("src/python/config.py").exists(),
    reason="Config file not found"
)
class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_pipeline(self):
        """Тест полного конвейера."""
        # Создаем временные данные
        temp_dir = tempfile.mkdtemp()
        try:
            # Создаем тестовый файл
            test_file = Path(temp_dir) / "test.md"
            test_file.write_text("# Тестовый документ\n\nЭто тестовое содержание для поиска.")
            
            # Инициализация компонентов
            from src.python.ingester import MarkdownIngester
            from src.python.embedder import Embedder
            from src.python.retriever import MojoRetriever
            
            ingester = MarkdownIngester()
            embedder = Embedder()
            retriever = MojoRetriever()
            
            # Парсинг
            chunks = ingester.parse_directory(temp_dir)
            assert len(chunks) > 0
            
            # Векторизация
            texts = [chunk.text for chunk in chunks]
            embeddings = embedder.encode(texts)
            assert embeddings.shape[0] == len(chunks)
            
            # Индексация
            metadata = [chunk.metadata for chunk in chunks]
            retriever.index_vectors(embeddings, metadata)
            
            # Поиск
            query_embedding = embedder.encode_single("тестовое содержание")
            results = retriever.search(query_embedding, k=3)
            
            assert len(results) > 0
            assert all("score" in result for result in results)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

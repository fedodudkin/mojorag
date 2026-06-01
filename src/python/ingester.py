"""
Парсинг и обработка документов.
Поддержка Markdown и текстовых файлов с извлечением метаданных.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Generator
from dataclasses import dataclass
import mistune
from datetime import datetime

try:
    import mojorag_core
    _MOJO_AVAILABLE = True
except ImportError:
    _MOJO_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Чанк документа с метаданными."""
    text: str
    metadata: Dict[str, Any]


class MarkdownIngester:
    """Парсер Markdown и текстовых файлов."""
    
    def __init__(self):
        self.parser = mistune.create_markdown(renderer="ast")
    
    def parse_directory(self, directory: str) -> List[DocumentChunk]:
        """
        Рекурсивно обходить директорию и парсить все .md и .txt файлы.
        
        Args:
            directory: Путь к директории с документами
            
        Returns:
            Список чанков документов
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Директория не существует: {directory}")
        
        chunks = []
        file_count = 0
        
        # Поиск всех поддерживаемых файлов
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".md", ".txt", ".markdown"}:
                try:
                    file_chunks = self.parse_file(file_path)
                    chunks.extend(file_chunks)
                    file_count += 1
                except Exception as e:
                    print(f"⚠️  Ошибка при парсинге файла {file_path}: {e}")
                    continue
        
        print(f"📁 Обработано {file_count} файлов, создано {len(chunks)} чанков")
        return chunks
    
    def parse_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Парсить отдельный файл и создавать чанки.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Список чанков из этого файла
        """
        # Чтение файла
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Пробуем другие кодировки
            try:
                with open(file_path, 'r', encoding='cp1251') as f:
                    content = f.read()
            except:
                raise ValueError(f"Не удалось прочитать файл {file_path} с поддерживаемыми кодировками")
        
        # Извлечение метаданных
        metadata = self._extract_metadata(file_path, content)
        
        # Очистка контента (удаление front matter если есть)
        clean_content = self._clean_content(content)
        
        # Разбиение на чанки
        chunks = self._chunk_text(clean_content, metadata)
        
        return chunks
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Извлекает метаданные из файла."""
        stat = file_path.stat()
        
        # Безопасное вычисление относительного пути
        try:
            source = str(file_path.resolve().relative_to(Path.cwd()))
        except ValueError:
            # Если файл вне cwd — используем абсолютный путь
            source = str(file_path.resolve())
        
        metadata = {
            "source": source,
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }
        
        # Попытка извлечь YAML front matter
        if content.startswith("---"):
            lines = content.split('\n')
            if len(lines) > 1 and lines[1].strip():
                try:
                    import yaml
                    front_matter_end = -1
                    for i, line in enumerate(lines[2:], 2):
                        if line.strip() == "---":
                            front_matter_end = i
                            break
                    
                    if front_matter_end > 0:
                        front_matter = '\n'.join(lines[2:front_matter_end])
                        yaml_data = yaml.safe_load(front_matter)
                        if isinstance(yaml_data, dict):
                            metadata.update(yaml_data)
                except ImportError:
                    pass  # yaml не установлен, пропускаем
                except:
                    pass  # Ошибка парсинга YAML, пропускаем
        
        # Извлечение заголовка из Markdown
        if "title" not in metadata:
            title = self._extract_title_from_markdown(content)
            if title:
                metadata["title"] = title
        
        return metadata
    
    def _extract_title_from_markdown(self, content: str) -> str:
        """Извлекает заголовок из Markdown контента."""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return ""
    
    def _clean_content(self, content: str) -> str:
        """Очищает контент от front matter и лишних элементов."""
        lines = content.split('\n')
        
        # Удаление YAML front matter
        if content.startswith("---"):
            front_matter_end = -1
            for i, line in enumerate(lines[2:], 2):
                if line.strip() == "---":
                    front_matter_end = i
                    break
            
            if front_matter_end > 0:
                content = '\n'.join(lines[front_matter_end + 1:])
        
        return content.strip()
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Разбивает текст на чанки.
        
        Args:
            text: Текст для разбивки
            metadata: Метаданные файла
            
        Returns:
            Список чанков
        """
        from .config import CHUNK_SIZE, CHUNK_OVERLAP
        
        if _MOJO_AVAILABLE:
            # Используем Mojo для высокопроизводительного чанкинга
            return self._chunk_with_mojo(text, metadata, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            # Fallback на Python реализацию
            return self._chunk_with_python(text, metadata, CHUNK_SIZE, CHUNK_OVERLAP)
    
    def _chunk_with_mojo(self, text: str, metadata: Dict[str, Any], chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """Чанкинг с использованием Mojo."""
        try:
            # Вызов Mojo функции
            mojo_chunks = mojorag_core.chunk_text(text, chunk_size, overlap)
            
            chunks = []
            for i, chunk_data in enumerate(mojo_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_start": chunk_data["start"],
                    "chunk_end": chunk_data["end"],
                    "chunk_length": len(chunk_data["text"]),
                })
                
                chunk = DocumentChunk(
                    text=chunk_data["text"],
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"⚠️  Ошибка Mojo чанкинга, использую Python fallback: {e}")
            return self._chunk_with_python(text, metadata, chunk_size, overlap)
    
    def _chunk_with_python(self, text: str, metadata: Dict[str, Any], chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """Python реализация чанкинга."""
        chunks = []
        
        if len(text) <= chunk_size:
            # Текст помещается в один чанк
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": 0,
                "chunk_start": 0,
                "chunk_end": len(text),
                "chunk_length": len(text),
            })
            
            chunk = DocumentChunk(text=text, metadata=chunk_metadata)
            chunks.append(chunk)
        else:
            # Разбиваем на пересекающиеся чанки
            position = 0
            chunk_index = 0
            
            while position < len(text):
                end_pos = min(position + chunk_size, len(text))
                
                # Ищем конец предложения или абзаца
                if end_pos < len(text):
                    # Ищем ближайший конец предложения
                    sentence_end = text.rfind('. ', position, end_pos)
                    if sentence_end > position + chunk_size // 2:
                        end_pos = sentence_end + 2
                    else:
                        # Ищем конец строки
                        line_end = text.rfind('\n', position, end_pos)
                        if line_end > position + chunk_size // 2:
                            end_pos = line_end + 1
                
                chunk_text = text[position:end_pos].strip()
                
                if chunk_text:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": chunk_index,
                        "chunk_start": position,
                        "chunk_end": end_pos,
                        "chunk_length": len(chunk_text),
                    })
                    
                    chunk = DocumentChunk(text=chunk_text, metadata=chunk_metadata)
                    chunks.append(chunk)
                
                # Сдвиг позиции
                position = end_pos - overlap
                chunk_index += 1
                
                # Защита от бесконечного цикла
                if position <= 0:
                    position = end_pos
        
        return chunks
    
    def get_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Возвращает статистику по обработанным чанкам."""
        if not chunks:
            return {"total_chunks": 0}
        
        total_chars = sum(len(chunk.text) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        
        # Статистика по источникам
        sources = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "average_chunk_size": avg_chunk_size,
            "sources": sources,
            "file_types": list(set(chunk.metadata.get("file_type", "unknown") for chunk in chunks)),
        }

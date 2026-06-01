"""
Конфигурация MojoRAG.
Автоопределение профиля железа и глобальные настройки.
"""

import os
import psutil
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# Профили оборудования
# =============================================================================


@dataclass
class HardwareProfile:
    """Профиль аппаратного обеспечения."""
    name: str
    llm_model: str
    llm_ctx: int
    llm_threads: int
    embed_batch_size: int
    description: str


def detect_profile() -> HardwareProfile:
    """
    Определяет оптимальный профиль на основе доступного железа.
    
    Returns:
        HardwareProfile с настройками под текущую машину.
    """
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_count = psutil.cpu_count(logical=True)
    threads = max(1, cpu_count - 2)  # Оставляем 2 потока системе
    
    if ram_gb >= 32:
        return HardwareProfile(
            name="performance",
            llm_model="Llama-3-8B-Instruct-q4.gguf",
            llm_ctx=8192,
            llm_threads=threads,
            embed_batch_size=128,
            description=f"Performance: {ram_gb:.1f} GB RAM, {cpu_count} cores"
        )
    elif ram_gb >= 16:
        return HardwareProfile(
            name="balanced",
            llm_model="phi-3-mini-4k-instruct-q4.gguf",
            llm_ctx=4096,
            llm_threads=threads,
            embed_batch_size=64,
            description=f"Balanced: {ram_gb:.1f} GB RAM, {cpu_count} cores"
        )
    else:
        return HardwareProfile(
            name="low_memory",
            llm_model="phi-3-mini-4k-instruct.Q4_0.gguf",
            llm_ctx=2048,
            llm_threads=max(1, cpu_count - 1),
            embed_batch_size=32,
            description=f"Low memory: {ram_gb:.1f} GB RAM, {cpu_count} cores"
        )


# =============================================================================
# Пути
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
MODELS_DIR = BASE_DIR / "models"
PROMPTS_DIR = BASE_DIR / "prompts"

# Автоматически создаём директории при импорте
for _dir in (DATA_DIR, INDEX_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Параметры чанкинга
# =============================================================================

CHUNK_SIZE = 512         # токенов
CHUNK_OVERLAP = 64       # токенов
MAX_CHUNKS_PER_QUERY = 5  # сколько чанков подавать в LLM


# =============================================================================
# Модель эмбеддингов
# =============================================================================

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIMENSION = 384
EMBED_BACKEND = "onnx"


# =============================================================================
# Окружение (переменные из .env)
# =============================================================================

def _get_env_or_default(key: str, default: str) -> str:
    """Читает переменную окружения или возвращает значение по умолчанию."""
    return os.environ.get(key, default)


# =============================================================================
# Активный профиль
# =============================================================================

_ACTIVE_PROFILE: HardwareProfile | None = None


def get_profile() -> HardwareProfile:
    """
    Возвращает активный профиль (ленивая инициализация).
    
    При первом вызове определяет профиль на основе переменной
    окружения MOJORAG_PROFILE или автоопределения.
    """
    global _ACTIVE_PROFILE
    if _ACTIVE_PROFILE is None:
        profile_name = _get_env_or_default("MOJORAG_PROFILE", "auto")
        
        if profile_name == "performance":
            _ACTIVE_PROFILE = HardwareProfile(
                name="performance",
                llm_model="Llama-3-8B-Instruct-q4.gguf",
                llm_ctx=8192,
                llm_threads=max(1, psutil.cpu_count(logical=True) - 2),
                embed_batch_size=128,
                description="Forced performance profile"
            )
        elif profile_name == "balanced":
            _ACTIVE_PROFILE = HardwareProfile(
                name="balanced",
                llm_model="phi-3-mini-4k-instruct-q4.gguf",
                llm_ctx=4096,
                llm_threads=max(1, psutil.cpu_count(logical=True) - 2),
                embed_batch_size=64,
                description="Forced balanced profile"
            )
        elif profile_name == "low_memory":
            _ACTIVE_PROFILE = HardwareProfile(
                name="low_memory",
                llm_model="phi-3-mini-4k-instruct.Q4_0.gguf",
                llm_ctx=2048,
                llm_threads=max(1, psutil.cpu_count(logical=True) - 1),
                embed_batch_size=32,
                description="Forced low memory profile"
            )
        else:
            _ACTIVE_PROFILE = detect_profile()
    
    return _ACTIVE_PROFILE


def get_model_path(model_name: str) -> Path:
    """Возвращает полный путь к модели."""
    return MODELS_DIR / model_name


def get_vector_index_path() -> Path:
    """Возвращает путь к файлу векторного индекса."""
    return INDEX_DIR / "vectors.bin"


def get_metadata_path() -> Path:
    """Возвращает путь к файлу метаданных."""
    return INDEX_DIR / "metadata.json"


# =============================================================================
# Валидация при импорте
# =============================================================================

def validate_environment() -> list[str]:
    """
    Проверяет готовность окружения.
    
    Returns:
        Список предупреждений (пустой список = всё OK).
    """
    warnings = []
    
    profile = get_profile()
    
    # Проверка наличия моделей
    llm_path = get_model_path(profile.llm_model)
    if not llm_path.exists():
        warnings.append(
            f"LLM модель не найдена: {llm_path}\n"
            f"Запустите: pixi run download-models"
        )
    
    # Проверка RAM
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if profile.name == "performance" and ram_gb < 32:
        warnings.append(
            f"Профиль 'performance' требует 32+ GB RAM, доступно {ram_gb:.1f} GB. "
            f"Рекомендуется переключиться на 'balanced'."
        )
    
    return warnings

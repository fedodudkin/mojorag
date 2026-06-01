"""
Скрипт загрузки моделей для MojoRAG.
Автономный — не зависит от других модулей проекта.
"""

import sys
from pathlib import Path

# Проверяем наличие huggingface_hub
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("Ошибка: нужен huggingface_hub. Установите: pip install huggingface-hub")
    sys.exit(1)

# Проверяем rich для прогресса
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    _RICH_AVAILABLE = True
    console = Console()
except ImportError:
    _RICH_AVAILABLE = False
    console = None

# Пути
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Загрузка моделей для MojoRAG")
print(f"Папка: {MODELS_DIR}")
print("=" * 60)


def _download_single(repo_id: str, filename: str, local_dir: str) -> None:
    """Скачать один файл."""
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )


def _download_snapshot(repo_id: str, local_dir: str) -> None:
    """Скачать репозиторий (snapshot)."""
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.bin", "*.safetensors", "pytorch_*"],
    )


def _download_with_spinner(description: str, fn, *args, **kwargs) -> None:
    """Загрузка со спиннером (rich) или без."""
    if _RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(description, total=None)
            fn(*args, **kwargs)
            progress.remove_task(task)
    else:
        print(f"  {description}...")
        fn(*args, **kwargs)


# 1. Эмбеддинг-модель (~90 MB)
print("\n[1/2] all-MiniLM-L6-v2 (ONNX)...")
try:
    _download_with_spinner(
        "Загрузка all-MiniLM-L6-v2...",
        _download_snapshot,
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=str(MODELS_DIR / "all-MiniLM-L6-v2"),
    )
    print("  Готово")
except Exception as e:
    print(f"  Ошибка: {e}")

# 2. LLM (~2.5 GB)
print("\n[2/2] Phi-3-mini-4k-instruct (q4)...")
try:
    _download_with_spinner(
        "Загрузка Phi-3-mini-4k-instruct-q4.gguf...",
        _download_single,
        repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
        filename="Phi-3-mini-4k-instruct-q4.gguf",
        local_dir=str(MODELS_DIR),
    )
    print("  Готово")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\n" + "=" * 60)
print("Загрузка завершена!")
print("=" * 60)
"""
MojoRAG - High-performance local RAG system using Mojo + Python
"""

__version__ = "0.1.0"
__author__ = "MojoRAG Team"

from .config import (
    BASE_DIR,
    DATA_DIR,
    INDEX_DIR,
    MODELS_DIR,
    PROMPTS_DIR,
    HardwareProfile,
    detect_profile,
)
from .orchestrator import MojoRAG

# Пытаемся импортировать скомпилированный модуль Mojo
try:

    import sys

    sys.path.insert(0, "/app/src/mojo")
    import mojorag_core

    _MOJO_AVAILABLE = True
except ImportError:
    _MOJO_AVAILABLE = False
    mojorag_core = None


def is_mojo_available() -> bool:
    """Check if Mojo module is available."""
    return _MOJO_AVAILABLE


__all__ = [
    "MojoRAG",
    "HardwareProfile",
    "detect_profile",
    "is_mojo_available",
    "BASE_DIR",
    "DATA_DIR",
    "INDEX_DIR",
    "MODELS_DIR",
    "PROMPTS_DIR",
]

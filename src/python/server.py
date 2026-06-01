"""
FastAPI сервер для MojoRAG.
REST API для индексации и вопрос-ответа.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from .config import get_profile, validate_environment
from .orchestrator import MojoRAG, create_mojorag

logger = logging.getLogger(__name__)


# Pydantic модели для запросов
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Вопрос пользователя")
    k: int = Field(default=5, ge=1, le=20, description="Количество релевантных чанков")
    strict: bool = Field(
        default=False, description="Строгий режим (только из контекста)"
    )
    stream: bool = Field(default=False, description="Стриминг ответа")


class IndexRequest(BaseModel):
    data_dir: str = Field(..., description="Директория с документами для индексации")


class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Ответ на вопрос")
    sources: List[Dict[str, Any]] = Field(
        default=[], description="Использованные источники"
    )
    stats: Dict[str, Any] = Field(default={}, description="Статистика поиска")


class IndexResponse(BaseModel):
    success: bool = Field(..., description="Успешность индексации")
    message: str = Field(..., description="Сообщение о результате")
    stats: Dict[str, Any] = Field(default={}, description="Статистика индексации")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус сервера")
    profile: Dict[str, Any] = Field(..., description="Профиль системы")
    index_loaded: bool = Field(..., description="Загружен ли индекс")
    warnings: List[str] = Field(default=[], description="Предупреждения")


# Глобальный экземпляр MojoRAG. 
# В продакшене лучше использовать более изолированное управление состоянием.
rag_instance: Optional[MojoRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global rag_instance
    print("Инициализация MojoRAG...")

    loop = asyncio.get_running_loop()
    rag_instance = await loop.run_in_executor(None, create_mojorag)

    print(f"MojoRAG инициализирован (профиль: {rag_instance.profile.name})")
    yield
    print("Завершение MojoRAG...")
    rag_instance = None


def create_app() -> FastAPI:
    """Создает FastAPI приложение."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI не установлен. Установите: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="MojoRAG API",
        description="Высокопроизводительная локальная RAG-система",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        # Безопасная сериализация ошибок
        errors = []
        for error in exc.errors():
            errors.append(
                {
                    "loc": str(error.get("loc", [])),
                    "msg": str(error.get("msg", "")),
                    "type": str(error.get("type", "")),
                }
            )
        return JSONResponse(status_code=400, content={"detail": errors})

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def get_app() -> FastAPI:
    """Получает или создает FastAPI приложение."""
    app = create_app()

    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Корневой эндпоинт."""
        return {"message": "MojoRAG API Server", "version": "0.1.0"}

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Проверка здоровья сервера."""
        global rag_instance

        # Проверка окружения
        warnings = validate_environment()

        # Информация о профиле
        profile = get_profile()
        profile_info = {
            "name": profile.name,
            "description": profile.description,
            "llm_model": profile.llm_model,
            "llm_ctx": profile.llm_ctx,
            "llm_threads": profile.llm_threads,
        }

        # Статус индекса
        index_loaded = rag_instance is not None and rag_instance._index_loaded

        return HealthResponse(
            status="healthy" if rag_instance else "initializing",
            profile=profile_info,
            index_loaded=index_loaded,
            warnings=warnings,
        )

    @app.post("/ask", response_model=QuestionResponse)
    async def ask_question(request: QuestionRequest):
        """Задать вопрос системе."""
        global rag_instance

        if not rag_instance:
            raise HTTPException(status_code=503, detail="Сервис не инициализирован")

        if not rag_instance._index_loaded:
            raise HTTPException(
                status_code=400,
                detail="Индекс не загружен. Сначала выполните индексацию",
            )

        try:
            # Поиск и генерация ответа
            if request.stream:
                # Стриминг не поддерживается в простом REST API
                raise HTTPException(
                    status_code=400,
                    detail="Стриминг не поддерживается в этом эндпоинте. Используйте WebSocket",
                )

            loop = asyncio.get_running_loop()

            # 1. Эмбеддим вопрос ОДИН раз
            query_vector = await loop.run_in_executor(
                None, rag_instance.embedder.encode_single, request.question
            )

            # 2. Ищем релевантные чанки
            retrieved = await loop.run_in_executor(
                None, rag_instance.retriever.search, query_vector, request.k
            )

            # 3. Генерируем ответ (передаём уже найденные чанки)
            answer = await loop.run_in_executor(
                None,
                lambda: "".join(
                    rag_instance.generator.generate(
                        question=request.question,
                        chunks=retrieved,
                        strict=request.strict,
                        stream=False,
                    )
                ),
            )

            return QuestionResponse(
                answer=answer,
                sources=[
                    {
                        "source": r.get("source", ""),
                        "score": r.get("score", 0.0),
                    }
                    for r in retrieved
                ],
                stats={
                    "question_length": len(request.question),
                    "sources_used": len(retrieved),
                    "response_length": len(answer),
                },
            )

        except Exception as e:
            logger.error(f"Ошибка обработки вопроса: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

    @app.post("/index")
    async def index_documents(request: IndexRequest):
        """Проиндексировать директорию с заметками."""
        if rag_instance is None:
            raise HTTPException(status_code=503, detail="RAG not initialized")

        loop = asyncio.get_running_loop()

        try:
            # Запускаем индексацию в отдельном потоке с таймаутом 5 минут
            stats = await asyncio.wait_for(
                loop.run_in_executor(None, rag_instance.ingest, str(request.data_dir)),
                timeout=300.0,  # 5 минут максимум
            )
            return {
                "status": "success",
                "chunks_indexed": stats.get("total_chunks", 0)
                if isinstance(stats, dict)
                else (stats if isinstance(stats, int) else 0),
            }
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504, detail="Индексация заняла более 5 минут"
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Ошибка индексации: {str(e)}")

    @app.get("/stats")
    async def get_stats():
        """Получить статистику системы."""
        global rag_instance

        if not rag_instance:
            raise HTTPException(status_code=503, detail="Сервис не инициализирован")

        try:
            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(None, rag_instance.get_stats)
            return stats
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            raise HTTPException(
                status_code=500, detail=f"Ошибка получения статистики: {str(e)}"
            )

    @app.post("/reload")
    async def reload_index():
        """Перезагрузить индекс."""
        global rag_instance

        if not rag_instance:
            raise HTTPException(status_code=503, detail="Сервис не инициализирован")

        try:
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(None, rag_instance.reload_index)
            return {
                "success": success,
                "message": "Индекс перезагружен"
                if success
                else "Ошибка перезагрузки индекса",
            }
        except Exception as e:
            logger.error(f"Ошибка перезагрузки индекса: {e}")
            raise HTTPException(
                status_code=500, detail=f"Ошибка перезагрузки: {str(e)}"
            )

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, **kwargs):
    """Запускает FastAPI сервер."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI не установлен. Установите: pip install fastapi uvicorn"
        )

    uvicorn.run("src.python.server:app", host=host, port=port, **kwargs)


# Создание FastAPI приложения
app = get_app()

if __name__ == "__main__":
    # Запуск сервера при прямом вызове
    logging.basicConfig(level=logging.INFO)
    run_server()

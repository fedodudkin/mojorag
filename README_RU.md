# MojoRAG

Высокопроизводительная локальная RAG-система на Mojo + Python

MojoRAG — это полностью локальная система Retrieval-Augmented Generation (RAG). Она использует **Mojo** для сверхбыстрого SIMD-оптимизированного векторного поиска и **Python** для оркестрации, интеграции ML-моделей и API-сервера. Все данные остаются на вашем устройстве.

## Возможности

- **Высокая производительность** — векторный поиск на Mojo с SIMD-оптимизацией (AVX-512)
- **Полностью локально** — без облачных сервисов, полная конфиденциальность данных
- **Умная индексация** — автоматический чанкинг текста с перекрытием и извлечением метаданных
- **Гибкие профили** — автоопределение оптимальных настроек по доступной RAM
- **Поддержка Markdown** — парсинг файлов `.md` и `.txt` с YAML frontmatter
- **REST API** — FastAPI-сервер с эндпоинтами `/health`, `/index`, `/ask`
- **CLI-интерфейс** — интерактивный и неинтерактивный CLI на базе Rich
- **Docker Ready** — полная контейнеризация с многоэтапной сборкой
- **Строгий режим** — ответы только из предоставленного контекста, без галлюцинаций

## Требования

- **Docker** (рекомендуется для всех платформ, включая Windows)
- **RAM**: минимум 16 ГБ (профиль balanced)
- **Диск**: ~5 ГБ свободного места (модели + индекс)

> **Пользователям Windows:** Mojo не работает нативно на Windows. Используйте Docker или WSL2.

## Быстрый старт

### 1. Клонирование и сборка

```bash
git clone <repo-url>
cd mojorag
make build
```

### 2. Загрузка моделей

```bash
python scripts/download_models.py
```

По умолчанию загружает модель для профиля balanced (Phi-3-mini-4k-instruct-q4, ~2.5 GB).
Для профилей performance/low_performance скачайте необходимые модели вручную.

### 3. Запуск сервера

```bash
make run
```

Сервер доступен по адресу `http://localhost:8000`. Первый запуск занимает 30–60 секунд на загрузку моделей.

### 4. Индексация документов

```bash
make index
```

Или через API:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"data_dir":"/app/data"}'
```

### 5. Задать вопрос

```bash
make ask
```

Или через API:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is MojoRAG?","k":3}'
```

## Использование

### REST API

| Метод | Эндпоинт | Описание |
|--------|----------|----------|
| `GET` | `/health` | Состояние сервера и индекса |
| `POST` | `/index` | Индексация документов из директории |
| `POST` | `/ask` | Задать вопрос и получить ответ |

### CLI

```bash
# Интерактивный режим
docker exec -it mojorag python3 -m src.python.main ask

# Неинтерактивный режим
docker exec mojorag python3 -m src.python.main ask --question "What is MojoRAG?"

# Информация о системе
docker exec mojorag python3 -m src.python.main info

# Индексация документов
docker exec mojorag python3 -m src.python.main index --data-dir /app/data
```

## Конфигурация

### Профили оборудования

| Профиль | RAM | LLM-модель | Контекст | Размер |
|---------|-----|-----------|----------|--------|
| `low_memory` | <16 ГБ | Phi-3-mini Q4_0 | 2048 | ~2.2 ГБ |
| `balanced` | 16–32 ГБ | Phi-3-mini Q4 | 4096 | ~2.5 ГБ |
| `performance` | 32+ ГБ | Llama-3-8B Q4_K_M | 8192 | ~5.5 ГБ |

Переопределение через переменную окружения:

```bash
export MOJORAG_PROFILE=balanced
```

### Параметры чанкинга

Отредактируйте `src/python/config.py`:

```python
CHUNK_SIZE = 512      # токенов на чанк
CHUNK_OVERLAP = 64    # перекрытие между чанками
MAX_CHUNKS_PER_QUERY = 5
```

## Docker

```bash
make build          # Собрать образ (с кэшем)
make build-no-cache # Собрать образ (без кэша)
make run            # Запустить контейнер
make stop           # Остановить контейнер
make logs           # Просмотр логов
make restart        # Перезапустить контейнер
make shell          # Открыть shell в контейнере
make clean-docker   # Удалить образы и кэш
```

## Тестирование

```bash
make test           # Запустить все тесты
make test-e2e       # Только E2E-тесты
```

## Структура проекта

```
mojorag/
├── data/                    # Документы пользователя (volume)
├── index/                   # Бинарные файлы индекса
├── models/                  # Загруженные модели (не в git)
├── prompts/                 # Шаблоны промптов Jinja2
│   ├── answer.j2
│   └── answer_strict.j2
├── scripts/
│   └── download_models.py   # Скрипт загрузки моделей
├── src/
│   ├── mojo/               # Исходный код Mojo (компилируется в .so)
│   │   ├── bindings.mojo    # Слой Python interop
│   │   ├── search.mojo      # Ядро векторного поиска
│   │   └── chunker.mojo    # Чанкинг текста
│   │       
│   └── python/             # Исходный код Python
│       ├── __init__.py
│       ├── config.py        # Конфигурация и профили оборудования
│       ├── main.py          # CLI (Click + Rich)
│       ├── ingester.py      # Парсинг Markdown
│       ├── embedder.py      # Генерация эмбеддингов (ONNX Runtime)
│       ├── retriever.py     # Поиск (Mojo + Python fallback)
│       ├── generator.py     # LLM-инференс (llama-cpp-python)
│       ├── orchestrator.py  # Центральный координатор
│       └── server.py        # FastAPI-сервер
├── tests/
│   ├── test_search.py       # Юнит-тесты
│   └── test_e2e.py          # Сквозные тесты
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .dockerignore
└── README.md
```

## Производительность

### Алгоритмическая сложность

| Компонент | Сложность | 10K чанков | 100K чанков |
|-----------|-----------|------------|-------------|
| Векторный поиск (brute-force) | O(N × dim) | ~15 мс | ~150 мс |
| Чанкинг текста | O(L) | <1 мс | <10 мс |
| Эмбеддинги (batch) | O(B × L × d²) | ~200 мс | ~200 мс |
| Генерация LLM | O(T × L) | 5–15 сек | 5–15 сек |

> Для коллекций свыше 50K чанков рассмотрите HNSW-индекс

## Решение проблем

**«Индекс не загружен» / "Index not loaded"**
Сначала выполните `POST /index`. Индекс хранится в памяти и должен быть пересобран после перезапуска контейнера.

**«Mojo module not available, using Python fallback»**
Mojo-модулю нужна библиотека `libKGENCompilerRTShared.so`. Пересоберите: `make build-no-cache`.

**«EOF when reading a line» в CLI**
Используйте флаг `--question` в неинтерактивном режиме:
```bash
docker exec mojorag python3 -m src.python.main ask --question "Your question"
```

**Нехватка памяти (OOM)**
Переключитесь на профиль `low_memory`: `export MOJORAG_PROFILE=low_memory`.

**Ошибки SSL при сборке**
Добавьте флаги `--trusted-host` в Dockerfile или используйте зеркало PyPI.


## Лицензия

MIT License — см. [LICENSE](LICENSE).

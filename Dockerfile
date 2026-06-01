# =============================================================================
# MojoRAG Dockerfile
# Двухэтапная сборка: компиляция Mojo -> лёгкий runtime
# Оптимизировано для Windows 10/WSL2 (linux/amd64)
# =============================================================================

# ---- Этап 1: Сборка Mojo-модуля ----
FROM --platform=linux/amd64 ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="/root/.pixi/bin:${PATH}"

# Проверка pixi
RUN pixi --version

# Создаём временный проект ТОЛЬКО для установки Mojo
RUN pixi init mojo-builder \
    -c https://conda.modular.com/max/ \
    -c conda-forge \
    && cd mojo-builder \
    && pixi add "mojo==0.26.2"

# Активация окружения
RUN cd /build/mojo-builder \
    && pixi shell-hook > /etc/profile.d/pixi.sh

WORKDIR /build/mojo-builder

# Копируем исходники Mojo
COPY src/mojo/ ./src/mojo/

# Компиляция Mojo-модуля в shared library
RUN set -eux; \
    mkdir -p compiled; \
    pixi run mojo build src/mojo/bindings.mojo \
        --emit shared-lib -o compiled/mojorag_core.so; \
    chmod 755 compiled/mojorag_core.so

# Копируем библиотеки, которые реально нужны mojorage_core.so (через ldd)
RUN set -eux; \
    PIXI_LIB=/build/mojo-builder/.pixi/envs/default/lib; \
    mkdir -p compiled/mojo_libs; \
    cp -P ${PIXI_LIB}/*.so* compiled/mojo_libs/ 2>/dev/null || true; \
    LD_LIBRARY_PATH=${PIXI_LIB} \
        ldd compiled/mojorag_core.so \
        | grep "=> /" | awk '{print $3}' \
        | xargs -I{} cp -vn {} compiled/mojo_libs/ 2>/dev/null || true; \
    echo "=== KGEN check ===" && find compiled/mojo_libs/ -name "*KGEN*" -ls; 

# ---- Этап 2: Runtime образ ----
FROM --platform=linux/amd64 ubuntu:22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src/mojo:/app \
    APP_HOME=/app \
    LD_LIBRARY_PATH=/app/mojo_libs:/usr/lib/x86_64-linux-gnu \
    MOJORAG_PROFILE=balanced

WORKDIR ${APP_HOME}

# Системные зависимости для runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    ca-certificates libstdc++6 libgomp1 \
    ninja-build cmake build-essential \
    && pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir --timeout 120 --retries 5 llama-cpp-python \
    && apt-get remove -y build-essential ninja-build cmake \
    && apt-get autoremove -y --purge \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt и устанавливаем Python-зависимости
COPY requirements.txt ${APP_HOME}/
RUN pip3 install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

COPY --from=builder /build/mojo-builder/compiled/mojorag_core.so /app/src/mojo/mojorag_core.so
COPY --from=builder /build/mojo-builder/compiled/mojo_libs/ /app/mojo_libs/

# Копируем Python-код
COPY src/python/ ${APP_HOME}/src/python/
COPY prompts/ ${APP_HOME}/prompts/
   
# Создаём директории для данных
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p ${APP_HOME}/data ${APP_HOME}/models ${APP_HOME}/index \
    && chown -R appuser:appuser ${APP_HOME}

# Переключаемся на непривилегированного пользователя
USER appuser

# Порт
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Точка входа — API сервер
CMD ["python3", "-m", "uvicorn", "src.python.server:app", "--host", "0.0.0.0", "--port", "8000"]

.PHONY: help build build-no-cache run stop restart logs shell test test-e2e index ask health download clean clean-docker quick-start

GREEN := \033[0;32m
NC := \033[0m

help: ## Показать это сообщение
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Сборка и запуск
build: ## Собрать Docker-образ (с кэшем)
	docker build --platform linux/amd64 -t mojorag:latest .

build-no-cache: ## Собрать Docker-образ (без кэша)
	docker build --platform linux/amd64 --no-cache -t mojorag:latest .

run: ## Запустить контейнер
	docker-compose up -d
	@sleep 5
	@curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "Сервер запускается..."

stop: ## Остановить контейнер
	docker-compose down

restart: ## Перезапустить контейнер
	docker-compose restart
	@sleep 5

logs: ## Показать логи в реальном времени
	docker-compose logs -f

shell: ## Открыть shell в контейнере
	docker exec -it mojorag bash

# Тестирование
test: ## Запустить все тесты
	docker exec mojorag python3 -m pytest /app/tests/ -v --tb=short

test-e2e: ## Запустить E2E тесты
	docker exec mojorag python3 -m pytest /app/tests/test_e2e.py -v --tb=short

# Работа с данными
download: ## Скачать модели
	python3 scripts/download_models.py --profile balanced

index: ## Проиндексировать документы (через API)
	curl -s -X POST http://localhost:8000/index -H "Content-Type: application/json" -d '{"data_dir":"/app/data"}' | python3 -m json.tool

ask: ## Задать вопрос (интерактивно)
	@read -p "Вопрос: " q; curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"$$q\",\"k\":3}" | python3 -m json.tool

health: ## Проверить здоровье сервера
	curl -s http://localhost:8000/health | python3 -m json.tool

# Очистка
clean: ## Очистить временные файлы
	rm -rf __mojocache__/
	rm -rf index/*.bin
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-docker: ## Очистить Docker-кэш и образы
	docker-compose down
	docker builder prune -af
	docker image prune -af

# Быстрый старт
quick-start: build run ## Сборка + запуск
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN) MojoRAG запущен на http://localhost:8000$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "Дальнейшие шаги:"
	@echo "  make download    — скачать модели"
	@echo "  make index       — проиндексировать документы"
	@echo "  make ask         — задать вопрос"
	@echo "  make test        — запустить тесты"

"""
MojoRAG CLI - Основная точка входа.
Интерактивная консоль с индексацией и вопрос-ответом.
"""

import click
import sys
import requests
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

try:
    from .config import get_profile, validate_environment, get_model_path
    from .orchestrator import MojoRAG
except ImportError:
    # Fallback для запуска как скрипта
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_profile, validate_environment, get_model_path
    from orchestrator import MojoRAG


console = Console()


def print_banner():
    """Выводит баннер приложения."""
    banner = """
[bold blue]MojoRAG[/bold blue] — Высокопроизводительная локальная RAG-система
[dim]Mojo + Python для максимальной скорости[/dim]
"""
    console.print(Panel(banner, border_style="blue"))


def print_profile_info():
    """Выводит информацию о текущем профиле."""
    profile = get_profile()
    
    table = Table(title="Аппаратный профиль", show_header=False)
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="green")
    
    table.add_row("Профиль", profile.name)
    table.add_row("LLM модель", Path(profile.llm_model).name)
    table.add_row("Контекст", f"{profile.llm_ctx} токенов")
    table.add_row("Потоки", str(profile.llm_threads))
    table.add_row("Batch size", str(profile.embed_batch_size))
    table.add_row("Описание", profile.description)
    
    console.print(table)


@click.group()
def cli():
    """MojoRAG - высокопроизводительная локальная RAG-система."""
    pass


@cli.command()
@click.option(
    "--data-dir", 
    type=click.Path(exists=True, path_type=Path), 
    help="Директория с документами для индексации"
)
def index(data_dir: Path):
    """Индексировать документы."""
    print_banner()
    
    # Проверка окружения
    warnings = validate_environment()
    if warnings:
        for warning in warnings:
            console.print(f"[yellow]⚠️  {warning}[/yellow]")
        console.print()
    
    print_profile_info()
    console.print()
    
    # Определение директории
    if not data_dir:
        from .config import DATA_DIR
        data_dir = DATA_DIR
    
    if not data_dir.exists():
        console.print(f"[red]❌ Директория не существует: {data_dir}[/red]")
        sys.exit(1)
    
    # Проверка наличия файлов
    md_files = list(data_dir.glob("*.md")) + list(data_dir.glob("*.txt"))
    if not md_files:
        console.print(f"[yellow]⚠️  В директории {data_dir} не найдено .md или .txt файлов[/yellow]")
        sys.exit(1)
    
    console.print(f"[green]📁 Найдено {len(md_files)} файлов для индексации[/green]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Инициализация MojoRAG...", total=None)
            
            rag = MojoRAG()
            progress.update(task, description="Индексация документов...")
            
            rag.ingest(str(data_dir))
            
            progress.update(task, description="Готово! ✅")
        
        console.print(f"[green]✅ Успешно проиндексировано {len(md_files)} документов[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка индексации: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--k", 
    default=5, 
    type=int, 
    help="Количество релевантных чанков"
)
@click.option(
    "--question", "-q",
    default=None,
    type=str,
    help="Вопрос (неинтерактивный режим)"
)
@click.option(
    "--strict",
    is_flag=True,
    help="Строгий режим (только из контекста)"
)
def ask(k: int, question: str | None, strict: bool):
    """Задать вопрос системе."""
    # Проверка что сервер жив
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code != 200:
            console.print("[red]❌ Сервер не отвечает. Запустите: make run[/red]")
            return
    except requests.exceptions.ConnectionError:
        console.print("[red]❌ Сервер не запущен. Запустите: make run[/red]")
        return

    # Неинтерактивный режим (флаг --question)
    if question:
        _ask_single(question, k, strict)
        return

    # Неинтерактивный режим (pipe)
    if not sys.stdin.isatty():
        question = sys.stdin.read().strip()
        if question:
            _ask_single(question, k, strict)
        else:
            console.print("[red]❌ Пустой вопрос[/red]")
        return

    # Интерактивный режим
    _ask_interactive(k, strict)


def _ask_single(question: str, k: int, strict: bool):
    """Отправить один вопрос через API."""
    console.print(f"[cyan]🤔 Вопрос: {question}[/cyan]")
    
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question, "k": k, "strict": strict},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print()
            console.print("[green]📝 Ответ:[/green]")
            console.print(data["answer"])
            if data.get("sources"):
                console.print(f"\n[dim]Источники: {len(data['sources'])}[/dim]")
        else:
            error_detail = response.json().get('detail', 'Неизвестная ошибка')
            console.print(f"[red]❌ {error_detail}[/red]")
    except requests.exceptions.Timeout:
        console.print("[red]❌ Таймаут (120 секунд)[/red]")
    except Exception as e:
        console.print(f"[red]❌ Ошибка: {e}[/red]")


def _ask_interactive(k: int, strict: bool):
    """Интерактивный режим с Prompt."""
    console.print("[green]💬 MojoRAG готов к вопросам![/green]")
    console.print("[dim]Введите 'quit' или 'exit' для выхода[/dim]")
    console.print()

    while True:
        try:
            question = Prompt.ask("[bold blue]Вопрос[/bold blue]")
            
            if question.lower() in ["quit", "exit", "выход"]:
                console.print("[green]👋 До свидания![/green]")
                break
            
            if not question.strip():
                continue

            console.print()
            _ask_single(question, k, strict)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹️  Прервано пользователем[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Ошибка: {e}[/red]")
            console.print()


@cli.command()
@click.option(
    "--host", 
    default="127.0.0.1", 
    help="Хост для API сервера"
)
@click.option(
    "--port", 
    default=8000, 
    type=int, 
    help="Порт для API сервера"
)
@click.option(
    "--profile", 
    type=click.Choice(["performance", "balanced", "low_memory", "auto"]),
    help="Принудительно выбрать профиль"
)
def serve(host: str, port: int, profile: str):
    """Запустить FastAPI сервер."""
    print_banner()
    
    # Установка профиля если указан
    if profile:
        import os
        os.environ["MOJORAG_PROFILE"] = profile
        console.print(f"[green]🔧 Установлен профиль: {profile}[/green]")
    
    # Проверка окружения
    warnings = validate_environment()
    if warnings:
        for warning in warnings:
            console.print(f"[yellow]⚠️  {warning}[/yellow]")
        console.print()
    
    # Проверка индекса
    from .config import get_vector_index_path
    index_path = get_vector_index_path()
    if not index_path.exists():
        console.print("[red]❌ Индекс не найден. Сначала выполните: mojorag index[/red]")
        sys.exit(1)
    
    print_profile_info()
    console.print()
    
    try:
        from .server import create_app
        import uvicorn
        
        app = create_app()
        
        console.print(f"[green]🚀 Запуск сервера на http://{host}:{port}[/green]")
        console.print("[dim]API docs: http://127.0.0.1:8000/docs[/dim]")
        console.print()
        
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        console.print("[red]❌ FastAPI не установлен. Установите: pip install fastapi uvicorn[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Ошибка запуска сервера: {e}[/red]")
        sys.exit(1)


@cli.command()
def info():
    """Показать информацию о системе."""
    print_banner()
    print_profile_info()
    
    # Проверка окружения
    warnings = validate_environment()
    console.print()
    
    if warnings:
        console.print("[yellow]⚠️  Предупреждения:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")
    else:
        console.print("[green]✅ Система готова к работе[/green]")


if __name__ == "__main__":
    cli()

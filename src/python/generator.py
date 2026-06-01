"""
LLM-инференс через llama-cpp-python.
Генерация ответов с использованием контекста из найденных чанков.
"""

from pathlib import Path
from typing import Generator
import threading

from llama_cpp import Llama
from jinja2 import Environment, FileSystemLoader

try:
    from .config import get_profile, get_model_path, PROMPTS_DIR
except ImportError:
    # Fallback для запуска как скрипта
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_profile, get_model_path, PROMPTS_DIR


class LLMGenerator:
    """Генератор ответов на основе LLM."""
    
    def __init__(self, profile=None):
        """
        Инициализация генератора.
        
        Args:
            profile: HardwareProfile или None (автоопределение).
        """
        self._profile = profile or get_profile()
        self._llm: Llama | None = None
        self._lock = threading.RLock()  
        self._jinja_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
        self._template = None
        self._strict_template = None
    
    @property
    def llm(self) -> Llama:
        """Ленивая загрузка LLM."""
        if self._llm is None:
            with self._lock:
                # Двойная проверка для потокобезопасности
                if self._llm is None:
                    model_path = get_model_path(self._profile.llm_model)
                    if not model_path.exists():
                        raise FileNotFoundError(f"Model not found: {model_path}")
                    
                    self._llm = Llama(
                        model_path=str(model_path),
                        n_ctx=self._profile.llm_ctx,
                        n_threads=self._profile.llm_threads,
                        n_batch=512,
                        use_mmap=True,
                        use_mlock=False,
                        verbose=False,
                    )
    
        return self._llm
    
    @property
    def template(self):
        """Основной шаблон ответа."""
        if self._template is None:
            self._template = self._jinja_env.get_template("answer.j2")
        return self._template
    
    @property
    def strict_template(self):
        """Строгий шаблон (только из контекста)."""
        if self._strict_template is None:
            self._strict_template = self._jinja_env.get_template("answer_strict.j2")
        return self._strict_template
    
    def _build_prompt(
        self,
        question: str,
        chunks: list[dict],
        strict: bool = False,
    ) -> str:
        """
        Формирует промпт для LLM из вопроса и контекста.
        
        Args:
            question: Вопрос пользователя.
            chunks: Список чанков с ключами 'text' и 'metadata'.
            strict: Использовать строгий шаблон (не додумывать).
            
        Returns:
            Отформатированный промпт.
        """
        template = self.strict_template if strict else self.template
        
        return template.render(
            question=question,
            retrieved_chunks=chunks,
        )
    
    def generate(
        self,
        question: str,
        chunks: list[dict],
        strict: bool = False,
        stream: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[str, None, None] | str:
        """
        Генерирует ответ с захватом блокировки для потокобезопасности.
        
        Args:
            question: Вопрос пользователя.
            chunks: Список релевантных чанков.
            strict: Использовать строгий режим.
            stream: Возвращать генератор токенов.
            max_tokens: Максимальное количество токенов.
            temperature: Температура генерации.
            
        Returns:
            Строка или генератор строк (если stream=True).
        """
        # Захватываем блокировку на всё время инференса
        with self._lock:
            prompt = self._build_prompt(question, chunks, strict=strict)
            
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            
            if stream:
                return self._stream_response(response)
            else:
                # Защита от бага llama-cpp-python (иногда возвращает генератор)
                if hasattr(response, '__iter__') and not isinstance(response, (str, dict)):
                    return ''.join(list(response))
                elif isinstance(response, dict):
                    return response["choices"][0]["message"]["content"]
                else:
                    return str(response)
    
    def _stream_response(self, response) -> Generator[str, None, None]:
        """
        Генератор токенов из стримингового ответа.
        
        Args:
            response: Итератор от llama-cpp-python.
            
        Yields:
            Токены ответа.
        """
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            if content:
                yield content
    
    def generate_batch(
        self,
        questions: list[str],
        chunks: list[list[dict]],
        strict: bool = False,
        max_tokens: int = 512,
    ) -> list[str]:
        """
        Генерирует ответы на несколько вопросов.
        
        Args:
            questions: Список вопросов.
            chunks: Список списков чанков для каждого вопроса.
            strict: Использовать строгий режим.
            max_tokens: Максимальное количество токенов на ответ.
            
        Returns:
            Список строк-ответов.
        """
        results = []
        for question, question_chunks in zip(questions, chunks):
            answer = self.generate(
                question=question,
                chunks=question_chunks,
                strict=strict,
                stream=False,
                max_tokens=max_tokens,
            )
            results.append(answer)
        
        return results
    
    def unload(self):
        """Выгружает модель из памяти."""
        if self._llm is not None:
            del self._llm
            self._llm = None
    
    def __del__(self):
        """Деструктор — выгружает модель."""
        self.unload()


def create_generator(profile_name: str | None = None):
    """
    Фабричная функция для создания LLMGenerator.
    
    Args:
        profile_name: Имя профиля или None для автоопределения.
        
    Returns:
        Экземпляр LLMGenerator.
    """
    from .config import HardwareProfile
    
    if profile_name:
        profile = HardwareProfile(
            name=profile_name,
            llm_model={
                "performance": "models/Llama-3-8B-Instruct-q4.gguf",
                "balanced": "models/phi-3-mini-4k-instruct-q4.gguf",
                "low_memory": "models/phi-3-mini-4k-instruct.Q4_0.gguf",
            }.get(profile_name, "models/phi-3-mini-4k-instruct-q4.gguf"),
            llm_ctx=8192 if profile_name == "performance" else 4096,
            llm_threads=8,
            embed_batch_size=64,
            description=f"Manual: {profile_name}",
        )
        return LLMGenerator(profile=profile)
    
    return LLMGenerator()

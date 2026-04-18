"""
LLM Provider Factory — supports OpenAI, Anthropic, Groq, Google Gemini, Ollama.
Usage:
    llm = get_llm("groq", "qwen-qwq-32b")
    llm = get_llm("openai", "gpt-4o")
"""

from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Factory function to instantiate an LLM from any supported provider.

    Args:
        provider: One of 'openai', 'anthropic', 'groq', 'google', 'ollama'.
                  Defaults to settings.default_llm_provider.
        model:    Model name/identifier. Defaults to settings.default_llm_model.
        temperature: Sampling temperature. Defaults to settings.default_temperature.
        max_tokens: Max output tokens. Defaults to settings.default_max_tokens.
        streaming: Enable streaming output.

    Returns:
        A LangChain BaseChatModel instance ready for use.
    """
    _provider = (provider or settings.default_llm_provider).lower()
    _model = model or settings.default_llm_model
    _temperature = temperature if temperature is not None else settings.default_temperature
    _max_tokens = max_tokens or settings.default_max_tokens

    logger.info(f"Initializing LLM: provider={_provider}, model={_model}")

    if _provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
            streaming=streaming,
            api_key=settings.openai_api_key,
        )

    elif _provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
            streaming=streaming,
            api_key=settings.anthropic_api_key,
        )

    elif _provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
            streaming=streaming,
            api_key=settings.groq_api_key,
        )

    elif _provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=_model,
            temperature=_temperature,
            max_output_tokens=_max_tokens,
            streaming=streaming,
            google_api_key=settings.google_api_key,
        )

    elif _provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=_model,
            temperature=_temperature,
            num_predict=_max_tokens,
            base_url=settings.ollama_base_url,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{_provider}'. "
            f"Choose from: openai, anthropic, groq, google, ollama"
        )


def get_default_llm(streaming: bool = False) -> BaseChatModel:
    """Return an LLM using the default provider/model from settings."""
    return get_llm(
        provider=settings.default_llm_provider,
        model=settings.default_llm_model,
        streaming=streaming,
    )


@lru_cache(maxsize=8)
def get_cached_llm(provider: str, model: str) -> BaseChatModel:
    """Return a cached (non-streaming) LLM instance. Useful for agents that share an LLM."""
    return get_llm(provider=provider, model=model)

from .base import AIProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .groq import GroqProvider


PROVIDERS: dict[str, AIProvider] = {
    "ollama": OllamaProvider(),
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "gemini": GeminiProvider(),
    "groq": GroqProvider(),
}


PROVIDER_METADATA = {
    "ollama": {
        "id": "ollama",
        "name": "Ollama",
        "requires_api_key": False,
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "requires_api_key": True,
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "requires_api_key": True,
    },
    "gemini": {
        "id": "gemini",
        "name": "Google Gemini",
        "requires_api_key": True,
    },
    "groq": {
        "id": "groq",
        "name": "Groq",
        "requires_api_key": True,
    },
}


def get_provider(provider: str | None) -> AIProvider:
    selected_provider = provider or "ollama"

    if selected_provider not in PROVIDERS:
        raise ValueError(
            f"Unsupported AI provider: {selected_provider}"
        )

    return PROVIDERS[selected_provider]


def get_provider_metadata() -> list[dict]:
    return [
        PROVIDER_METADATA[provider_id]
        for provider_id in PROVIDERS
        if provider_id in PROVIDER_METADATA
    ]
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


def get_provider(provider: str | None) -> AIProvider:
    selected_provider = provider or "ollama"

    if selected_provider not in PROVIDERS:
        raise ValueError(
            f"Unsupported AI provider: {selected_provider}"
        )

    return PROVIDERS[selected_provider]
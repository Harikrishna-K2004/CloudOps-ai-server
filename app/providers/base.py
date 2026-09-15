from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderErrorType(str, Enum):
    INVALID_API_KEY = "invalid_api_key"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_NOT_FOUND = "model_not_found"
    ACCESS_DENIED = "access_denied"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    # Provider-specific metadata that must survive
    # the tool-calling round trip.
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class AIResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(
        default_factory=list
    )


class AIProvider(ABC):

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        api_key: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AIResponse:
        pass

    async def generate(
        self,
        message: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> str:
        response = await self.chat(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model=model,
            api_key=api_key,
        )

        return response.content

    @abstractmethod
    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        pass
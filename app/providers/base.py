from abc import ABC, abstractmethod
from enum import Enum


class ProviderErrorType(str, Enum):
    INVALID_API_KEY = "invalid_api_key"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_NOT_FOUND = "model_not_found"
    ACCESS_DENIED = "access_denied"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    UNKNOWN = "unknown"


class AIProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        message: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        pass
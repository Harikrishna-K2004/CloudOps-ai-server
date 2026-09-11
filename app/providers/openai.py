from openai import AsyncOpenAI, APIStatusError
from .base import AIProvider, ProviderErrorType


class OpenAIProvider(AIProvider):

    async def generate(
        self,
        message: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> str:
        if not api_key:
            raise ValueError(
                "OpenAI API key is required"
            )

        client = AsyncOpenAI(api_key=api_key)

        try:
            response = await client.responses.create(
                model=model or "gpt-4.1-mini",
                input=message,
            )

            return response.output_text

        except APIStatusError as error:
            if error.status_code in (401, 403):
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "OpenAI API key is invalid or unauthorized"
                )

            if error.status_code == 429:
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "OpenAI rate limit or quota exceeded"
                )

            if error.status_code == 404:
                raise ValueError(
                    f"{ProviderErrorType.MODEL_NOT_FOUND.value}: "
                    "The selected OpenAI model was not found or is unavailable"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "OpenAI request failed"
            )

    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        if not api_key:
            raise ValueError(
                "OpenAI API key is required"
            )

        client = AsyncOpenAI(api_key=api_key)

        try:
            models = await client.models.list()

            return [
                {
                    "id": model.id,
                    "name": model.id,
                }
                for model in models.data
            ]

        except APIStatusError as error:
            if error.status_code in (401, 403):
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "OpenAI API key is invalid or unauthorized"
                )

            if error.status_code == 429:
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "OpenAI rate limit or quota exceeded"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Failed to fetch OpenAI models"
            )
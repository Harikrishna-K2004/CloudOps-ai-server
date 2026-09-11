from anthropic import AsyncAnthropic, APIStatusError
from .base import AIProvider, ProviderErrorType


class AnthropicProvider(AIProvider):

    async def generate(
        self,
        message: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> str:
        if not api_key:
            raise ValueError(
                "Anthropic API key is required"
            )

        client = AsyncAnthropic(api_key=api_key)

        try:
            response = await client.messages.create(
                model=model or "claude-sonnet-4-5",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
            )

            return "".join(
                block.text
                for block in response.content
                if hasattr(block, "text")
            )

        except APIStatusError as error:
            if error.status_code in (401, 403):
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "Anthropic API key is invalid or unauthorized"
                )

            if error.status_code == 429:
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "Anthropic rate limit or quota exceeded"
                )

            if error.status_code == 404:
                raise ValueError(
                    f"{ProviderErrorType.MODEL_NOT_FOUND.value}: "
                    "The selected Anthropic model was not found or is unavailable"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Anthropic request failed"
            )

    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        if not api_key:
            raise ValueError(
                "Anthropic API key is required"
            )

        client = AsyncAnthropic(api_key=api_key)

        try:
            models = await client.models.list()

            return [
                {
                    "id": model.id,
                    "name": model.display_name or model.id,
                }
                for model in models.data
            ]

        except APIStatusError as error:
            if error.status_code in (401, 403):
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "Anthropic API key is invalid or unauthorized"
                )

            if error.status_code == 429:
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "Anthropic rate limit or quota exceeded"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Failed to fetch Anthropic models"
            )
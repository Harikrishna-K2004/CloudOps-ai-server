from groq import AsyncGroq, APIStatusError
from .base import AIProvider, ProviderErrorType
from app.model_catalog.groq import is_groq_chat_model

class GroqProvider(AIProvider):

    async def generate(
        self,
        message: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> str:
        if not api_key:
            raise ValueError(
                "Groq API key is required"
            )

        client = AsyncGroq(api_key=api_key)

        try:
            response = await client.chat.completions.create(
                model=model or "openai/gpt-oss-20b",
                messages=[{"role": "user", "content": message}],
            )

        except APIStatusError as error:
            if error.status_code in (401, 403):
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "Groq API key is invalid or unauthorized"
                )

            if error.status_code == 429:
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "Groq rate limit or quota exceeded"
                )

            if error.status_code == 404:
                raise ValueError(
                    f"{ProviderErrorType.MODEL_NOT_FOUND.value}: "
                    "The selected Groq model was not found or is unavailable"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Groq request failed"
            )

        content = response.choices[0].message.content or ""

        if "<think>" in content and "</think>" in content:
            content = content.split("</think>", 1)[1].strip()

        return content

    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        if not api_key:
            raise ValueError("Groq API key is required")

        client = AsyncGroq(api_key=api_key)

        try:
            models = await client.models.list()

        except APIStatusError as error:
            if error.status_code in (401, 403):
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "Groq API key is invalid or unauthorized"
                )

            if error.status_code == 429:
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "Groq rate limit or quota exceeded"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Failed to fetch Groq models"
            )

        return [
            {
                "id": model.id,
                "name": model.id,
                "owned_by": getattr(model, "owned_by", None),
            }
            for model in models.data
            if is_groq_chat_model(model.id)
        ]
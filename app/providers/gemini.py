from google import genai
from .base import AIProvider, ProviderErrorType


class GeminiProvider(AIProvider):

    async def generate(
        self,
        message: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> str:
        if not api_key:
            raise ValueError(
                "Gemini API key is required"
            )

        client = genai.Client(api_key=api_key)

        try:
            response = await client.aio.models.generate_content(
                model=model or "gemini-2.5-flash",
                contents=message,
            )

            return response.text or ""

        except Exception as error:
            error_message = str(error).lower()

            if "401" in error_message or "unauthorized" in error_message:
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "Gemini API key is invalid or unauthorized"
                )

            if "403" in error_message or "permission" in error_message:
                raise ValueError(
                    f"{ProviderErrorType.ACCESS_DENIED.value}: "
                    "Access to the Gemini model was denied"
                )

            if "429" in error_message or "quota" in error_message:
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "Gemini rate limit or quota exceeded"
                )

            if "404" in error_message or "not found" in error_message:
                raise ValueError(
                    f"{ProviderErrorType.MODEL_NOT_FOUND.value}: "
                    "The selected Gemini model was not found or is unavailable"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Gemini request failed"
            )

    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        if not api_key:
            raise ValueError(
                "Gemini API key is required"
            )

        client = genai.Client(api_key=api_key)

        try:
            models = []

            async for model in await client.aio.models.list():
                if model.name:
                    model_id = model.name.split("/")[-1]

                    models.append(
                        {
                            "id": model_id,
                            "name": model.display_name or model_id,
                        }
                    )

            return models

        except Exception as error:
            error_message = str(error).lower()

            if "401" in error_message or "unauthorized" in error_message:
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "Gemini API key is invalid or unauthorized"
                )

            if "403" in error_message or "permission" in error_message:
                raise ValueError(
                    f"{ProviderErrorType.ACCESS_DENIED.value}: "
                    "Access to Gemini models was denied"
                )

            if "429" in error_message or "quota" in error_message:
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "Gemini rate limit or quota exceeded"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Failed to fetch Gemini models"
            )
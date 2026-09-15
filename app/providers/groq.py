import json

from groq import AsyncGroq, APIStatusError

from .base import (
    AIProvider,
    AIResponse,
    ProviderErrorType,
    ToolCall,
)
from app.model_catalog.groq import is_groq_chat_model


class GroqProvider(AIProvider):

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        api_key: str | None = None,
        tools: list[dict] | None = None,
    ) -> AIResponse:
        if not api_key:
            raise ValueError(
                "Groq API key is required"
            )

        client = AsyncGroq(api_key=api_key)

        try:
            request: dict = {
                "model": model or "openai/gpt-oss-20b",
                "messages": messages,
            }

            if tools:
                request["tools"] = tools

            response = await client.chat.completions.create(
                **request
            )

            message = response.choices[0].message

            tool_calls: list[ToolCall] = []

            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(
                        call.function.arguments
                    )
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=call.id,
                        name=call.function.name,
                        arguments=arguments,
                    )
                )

            content = message.content or ""

            if "<think>" in content and "</think>" in content:
                content = content.split(
                    "</think>",
                    1,
                )[1].strip()

            return AIResponse(
                content=content,
                tool_calls=tool_calls,
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
                f"Groq request failed: {error.body}"
            )

    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        if not api_key:
            raise ValueError(
                "Groq API key is required"
            )

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
                "owned_by": getattr(
                    model,
                    "owned_by",
                    None,
                ),
            }
            for model in models.data
            if is_groq_chat_model(model.id)
        ]
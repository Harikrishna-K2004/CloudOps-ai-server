import json

from anthropic import AsyncAnthropic, APIStatusError

from .base import (
    AIProvider,
    AIResponse,
    ProviderErrorType,
    ToolCall,
)


class AnthropicProvider(AIProvider):

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        api_key: str | None = None,
        tools: list[dict] | None = None,
    ) -> AIResponse:
        if not api_key:
            raise ValueError(
                "Anthropic API key is required"
            )

        client = AsyncAnthropic(api_key=api_key)

        anthropic_messages: list[dict] = []

        for message in messages:
            role = message["role"]

            if role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message[
                                    "tool_call_id"
                                ],
                                "content": message[
                                    "content"
                                ],
                            }
                        ],
                    }
                )
                continue

            if role == "assistant" and message.get(
                "tool_calls"
            ):
                content: list[dict] = []

                if message.get("content"):
                    content.append(
                        {
                            "type": "text",
                            "text": message["content"],
                        }
                    )

                for call in message["tool_calls"]:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": json.loads(
                                call["function"]["arguments"]
                            ),
                        }
                    )

                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
                continue

            anthropic_messages.append(
                {
                    "role": role,
                    "content": message.get(
                        "content",
                        "",
                    ),
                }
            )

        anthropic_tools: list[dict] = []

        for tool in tools or []:
            function = tool.get(
                "function",
                {},
            )

            anthropic_tools.append(
                {
                    "name": function["name"],
                    "description": function.get(
                        "description",
                        "",
                    ),
                    "input_schema": function.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                }
            )

        try:
            request: dict = {
                "model": model or "claude-sonnet-4-5",
                "max_tokens": 4096,
                "messages": anthropic_messages,
            }

            if anthropic_tools:
                request["tools"] = anthropic_tools

            response = await client.messages.create(
                **request
            )

            tool_calls: list[ToolCall] = []
            text_parts: list[str] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(
                        block.text
                    )

                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

            return AIResponse(
                content="".join(text_parts),
                tool_calls=tool_calls,
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
import json

from google import genai
from google.genai import types

from .base import (
    AIProvider,
    AIResponse,
    ProviderErrorType,
    ToolCall,
)


class GeminiProvider(AIProvider):

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        api_key: str | None = None,
        tools: list[dict] | None = None,
    ) -> AIResponse:
        if not api_key:
            raise ValueError(
                "Gemini API key is required"
            )

        client = genai.Client(
            api_key=api_key
        )

        contents: list[dict] = []

        for message in messages:
            role = message["role"]

            if role == "user":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": message.get(
                                    "content",
                                    "",
                                )
                            }
                        ],
                    }
                )

            elif role == "assistant":
                parts: list[dict] = []

                if message.get("content"):
                    parts.append(
                        {
                            "text": message["content"]
                        }
                    )

                for call in message.get(
                    "tool_calls",
                    [],
                ):
                    function_call = {
                        "name": call[
                            "function"
                        ]["name"],
                        "args": json.loads(
                            call[
                                "function"
                            ]["arguments"]
                        ),
                    }

                    thought_signature = call.get("metadata", {}).get("thought_signature")

                    part = {"function_call": function_call}

                    if thought_signature:
                        part["thought_signature"] = thought_signature

                    parts.append(part)

                contents.append(
                    {
                        "role": "model",
                        "parts": parts,
                    }
                )

            elif role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": message[
                                        "name"
                                    ],
                                    "response": {
                                        "result": json.loads(
                                            message[
                                                "content"
                                            ]
                                        )
                                    },
                                }
                            }
                        ],
                    }
                )

        declarations: list[
            types.FunctionDeclaration
        ] = []

        for tool in tools or []:
            function = tool.get(
                "function",
                {},
            )

            declarations.append(
                types.FunctionDeclaration(
                    name=function["name"],
                    description=function.get(
                        "description",
                        "",
                    ),
                    parameters_json_schema=function.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                )
            )

        config = None

        if declarations:
            config = types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        function_declarations=declarations
                    )
                ]
            )

        try:
            response = await client.aio.models.generate_content(
                model=model or "gemini-2.5-flash",
                contents=contents,
                config=config,
            )

            tool_calls: list[ToolCall] = []
            text_parts: list[str] = []

            candidate = (
                response.candidates[0]
                if response.candidates
                else None
            )

            if candidate:
                for part in candidate.content.parts:

                    if getattr(
                        part,
                        "text",
                        None,
                    ):
                        text_parts.append(
                            part.text
                        )

                    function_call = getattr(
                        part,
                        "function_call",
                        None,
                    )

                    if function_call:
                        metadata: dict = {}

                        thought_signature = getattr(
                            part,
                            "thought_signature",
                            None,
                        )

                        if thought_signature:
                            metadata[
                                "thought_signature"
                            ] = thought_signature

                        tool_calls.append(
                            ToolCall(
                                id=(
                                    f"gemini-call-"
                                    f"{len(tool_calls)}"
                                ),
                                name=function_call.name,
                                arguments=dict(
                                    function_call.args
                                ),
                                metadata=metadata,
                            )
                        )

            return AIResponse(
                content="".join(text_parts),
                tool_calls=tool_calls,
            )

        except Exception as error:
            error_message = str(error).lower()

            if (
                "401" in error_message
                or "unauthorized" in error_message
            ):
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "Gemini API key is invalid or unauthorized"
                )

            if (
                "403" in error_message
                or "permission" in error_message
            ):
                raise ValueError(
                    f"{ProviderErrorType.ACCESS_DENIED.value}: "
                    "Access to the Gemini model was denied"
                )

            if (
                "429" in error_message
                or "quota" in error_message
            ):
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "Gemini rate limit or quota exceeded"
                )

            if (
                "404" in error_message
                or "not found" in error_message
            ):
                raise ValueError(
                    f"{ProviderErrorType.MODEL_NOT_FOUND.value}: "
                    "The selected Gemini model was not found or is unavailable"
                )

            print(
                "Gemini API error:",
                repr(error),
            )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                f"Gemini request failed: {error}"
            )

    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        if not api_key:
            raise ValueError(
                "Gemini API key is required"
            )

        client = genai.Client(
            api_key=api_key
        )

        try:
            models = []

            async for model in await client.aio.models.list():
                if model.name:
                    model_id = model.name.split(
                        "/"
                    )[-1]

                    models.append(
                        {
                            "id": model_id,
                            "name": (
                                model.display_name
                                or model_id
                            ),
                        }
                    )

            return models

        except Exception as error:
            error_message = str(error).lower()

            if (
                "401" in error_message
                or "unauthorized" in error_message
            ):
                raise ValueError(
                    f"{ProviderErrorType.INVALID_API_KEY.value}: "
                    "Gemini API key is invalid or unauthorized"
                )

            if (
                "403" in error_message
                or "permission" in error_message
            ):
                raise ValueError(
                    f"{ProviderErrorType.ACCESS_DENIED.value}: "
                    "Access to Gemini models was denied"
                )

            if (
                "429" in error_message
                or "quota" in error_message
            ):
                raise ValueError(
                    f"{ProviderErrorType.RATE_LIMIT.value}: "
                    "Gemini rate limit or quota exceeded"
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Failed to fetch Gemini models"
            )
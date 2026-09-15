import httpx

from .base import (
    AIProvider,
    AIResponse,
    ProviderErrorType,
    ToolCall,
)


class OllamaProvider(AIProvider):

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        api_key: str | None = None,
        tools: list[dict] | None = None,
    ) -> AIResponse:
        selected_model = model or "qwen3"

        payload: dict = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools

        # Ollama may take several minutes to generate a response,
        # especially when running large models locally.
        #
        # We therefore use:
        # - finite connection/write/pool timeouts
        # - unlimited read timeout
        #
        # This prevents CloudOps from treating a slow model
        # generation as an unavailable Ollama server.
        timeout = httpx.Timeout(
            connect=30.0,
            read=None,
            write=30.0,
            pool=30.0,
        )

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
            ) as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()
                message = data.get("message", {})

                tool_calls: list[ToolCall] = []

                for index, call in enumerate(
                    message.get("tool_calls", [])
                ):
                    function = call.get(
                        "function",
                        {},
                    )

                    tool_calls.append(
                        ToolCall(
                            id=f"ollama-call-{index}",
                            name=function.get(
                                "name",
                                "",
                            ),
                            arguments=function.get(
                                "arguments",
                                {},
                            ),
                        )
                    )

                content = message.get(
                    "content",
                    "",
                )

                return AIResponse(
                    content=content,
                    tool_calls=tool_calls,
                )

        except httpx.ConnectError:
            raise ValueError(
                f"{ProviderErrorType.PROVIDER_UNAVAILABLE.value}: "
                "Ollama is not running. "
                "Please start Ollama and try again."
            )

        except httpx.TimeoutException:
            raise ValueError(
                f"{ProviderErrorType.PROVIDER_UNAVAILABLE.value}: "
                "Unable to connect to Ollama. "
                "Please make sure Ollama is running."
            )

        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                raise ValueError(
                    f"{ProviderErrorType.MODEL_NOT_FOUND.value}: "
                    "The selected Ollama model was not found."
                )

            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Ollama request failed."
            )

    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=5.0,
                    write=5.0,
                    pool=5.0,
                ),
            ) as client:
                response = await client.get(
                    "http://localhost:11434/api/tags",
                )

                response.raise_for_status()

                data = response.json()

                return [
                    {
                        "id": model["name"],
                        "name": model["name"],
                    }
                    for model in data.get(
                        "models",
                        [],
                    )
                ]

        except httpx.ConnectError:
            raise ValueError(
                f"{ProviderErrorType.PROVIDER_UNAVAILABLE.value}: "
                "Ollama is not running. "
                "Please start Ollama and try again."
            )

        except httpx.TimeoutException:
            raise ValueError(
                f"{ProviderErrorType.PROVIDER_UNAVAILABLE.value}: "
                "Ollama is not responding. "
                "Please try again."
            )

        except httpx.HTTPStatusError:
            raise ValueError(
                f"{ProviderErrorType.UNKNOWN.value}: "
                "Ollama model discovery failed."
            )
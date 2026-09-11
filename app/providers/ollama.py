import httpx
from .base import AIProvider


class OllamaProvider(AIProvider):

    async def generate(
        self,
        message: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> str:
        selected_model = model or "qwen3"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": selected_model,
                        "prompt": message,
                        "stream": False,
                    },
                    timeout=120,
                )

                response.raise_for_status()
                data = response.json()

                return data["response"]

        except httpx.ConnectError:
            raise ValueError(
                "provider_unavailable: Ollama is not running. "
                "Please start Ollama and try again."
            )

        except httpx.TimeoutException:
            raise ValueError(
                "provider_unavailable: Ollama is not responding. "
                "Please try again."
            )

        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                raise ValueError(
                    "model_not_found: The selected Ollama model was not found."
                )

            raise ValueError(
                "unknown: Ollama request failed."
            )

    async def get_models(
        self,
        api_key: str | None = None,
    ) -> list[dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:11434/api/tags",
                    timeout=5,
                )
                response.raise_for_status()

                data = response.json()

                return [
                    {"id": model["name"], "name": model["name"]}
                    for model in data.get("models", [])
                ]

        except httpx.ConnectError:
            raise ValueError(
                "provider_unavailable: Ollama is not running. "
                "Please start Ollama and try again."
            )

        except httpx.TimeoutException:
            raise ValueError(
                "provider_unavailable: Ollama is not responding. "
                "Please try again."
            )

        except httpx.HTTPStatusError:
            raise ValueError(
                "unknown: Ollama model discovery failed."
            )
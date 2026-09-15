from aiohttp import request
import json

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.providers import (
    get_provider,
    get_provider_metadata,
)
from app.tools.loader import discover_integrations
from app.tools.manager import (
    get_selected_tools,
    get_user_integration_catalog,
    build_selected_integration_tools,
    close_user_tools,
)
from app.tools.selector import select_integrations
from app.tools.tool_selector import select_tools
from app.tools.registry import execute_tool, get_model_tools
from app.context.routes import router as context_router


app = FastAPI(title="CloudOps AI Server")

app.include_router(context_router)
discover_integrations()


MAX_TOOL_ROUNDS = 10


class ChatRequest(BaseModel):
    userId: str
    message: str
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None

    tool_credentials: dict[
        str,
        dict[str, object],
    ] = {}

    context: list[dict] = []


class TitleRequest(BaseModel):
    userId: str
    message: str
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ai-server",
    }


@app.get("/api/providers")
async def get_providers():
    return {
        "providers": get_provider_metadata()
    }


@app.get("/api/models/{provider_name}")
async def get_models(
    provider_name: str,
    authorization: str | None = Header(
        default=None
    ),
):
    try:
        api_key = None

        if authorization and authorization.startswith(
            "Bearer "
        ):
            api_key = authorization[7:]

        provider = get_provider(
            provider_name
        )

        models = await provider.get_models(
            api_key=api_key,
        )

        return {
            "provider": provider_name,
            "models": models,
        }

    except ValueError as error:
        return {
            "error": str(error),
        }

    except Exception as error:
        print(
            f"Model discovery error: {error}"
        )

        return {
            "error": "Failed to fetch models",
        }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    tools = []
    selected_tools = []
    connections = []

    try:
        provider = get_provider(
            request.provider
        )

        # ==================================================
        # STAGE 1
        # Select relevant integration(s)
        # ==================================================

        integration_catalog = (
            get_user_integration_catalog(
                tool_credentials=(
                    request.tool_credentials
                ),
            )
        )

        selected_integrations = (
            await select_integrations(
                provider=provider,
                message=request.message,
                integrations=integration_catalog,
                model=request.model,
                api_key=request.api_key,
            )
        )

        print(
            "Stage 1 selected integrations:",
            selected_integrations,
        )

        # ==================================================
        # Context
        # ==================================================

        context_text = ""

        if request.context:
            print("RETRIEVED CONTEXT:", request.context)

            context_parts = []

            for item in request.context:
                content = item.get("content", "")
                if not content:
                    continue

                metadata = item.get("metadata", {}) or {}
                role = metadata.get("role", "unknown")

                if role == "user":
                    label = "USER"
                elif role == "assistant":
                    label = "ASSISTANT"
                else:
                    label = "CONTEXT"

                context_parts.append(
                    f"[{label}]\n{content}"
                )

            context_text = "\n\n".join(context_parts)

        user_content = request.message

        if context_text:
            user_content = (
                "Relevant context from this conversation is provided below.\n\n"
                "Important rules:\n"
                "- Treat statements labeled [USER] as statements made by the user.\n"
                "- Treat statements labeled [ASSISTANT] as previous assistant responses, "
                "which may be incorrect and must not be treated as authoritative facts.\n"
                "- Use the relevant [USER] information to answer the current request.\n\n"
                f"{context_text}\n\n"
                "Current user request:\n\n"
                f"{request.message}"
            )

        messages: list[dict] = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        # ==================================================
        # No integration required
        # ==================================================

        if not selected_integrations:
            response = await provider.chat(
                messages=messages,
                model=request.model,
                api_key=request.api_key,
                tools=None,
            )

            return {
                "response": response.content,
            }

        # ==================================================
        # Discover ONLY selected integration(s)
        # ==================================================

        tools, connections = (
            await build_selected_integration_tools(
                selected_provider_ids=(
                    selected_integrations
                ),
                tool_credentials=(
                    request.tool_credentials
                ),
            )
        )

        # ==================================================
        # STAGE 2
        # Select relevant tool(s)
        # ==================================================

        selected_tool_ids = await select_tools(
            provider=provider,
            message=request.message,
            tools=tools,
            model=request.model,
            api_key=request.api_key,
        )

        print(
            "Stage 2 selected tools:",
            selected_tool_ids,
        )

        # ==================================================
        # STAGE 3
        # Expose full definitions ONLY for selected tools
        # ==================================================

        selected_tools = get_selected_tools(
            tools=tools,
            selected_tool_ids=selected_tool_ids,
        )

        print(
            "Stage 3 tools exposed to model:",
            [
                tool.id
                for tool in selected_tools
            ],
        )

        model_tools = get_model_tools(
            selected_tools
        )

        # ==================================================
        # Tool-calling loop
        # ==================================================

        for _ in range(MAX_TOOL_ROUNDS):
            response = await provider.chat(
                messages=messages,
                model=request.model,
                api_key=request.api_key,
                tools=model_tools or None,
            )

            if not response.tool_calls:
                return {
                    "response": response.content,
                }

            assistant_tool_calls = []

            for tool_call in response.tool_calls:
                assistant_tool_call = {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(
                            tool_call.arguments
                        ),
                    },
                }

                # Preserve provider-specific metadata such
                # as Gemini's thought_signature.
                if tool_call.metadata:
                    assistant_tool_call[
                        "metadata"
                    ] = tool_call.metadata

                assistant_tool_calls.append(
                    assistant_tool_call
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": assistant_tool_calls,
                }
            )

            for tool_call in response.tool_calls:
                result = await execute_tool(
                    tool_id=_resolve_tool_id(
                        selected_tools,
                        tool_call.name,
                    ),
                    arguments=tool_call.arguments,
                    user_id=request.userId,
                    tools=selected_tools,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(
                            result,
                            default=str,
                        ),
                    }
                )

        raise HTTPException(
            status_code=502,
            detail=(
                "AI tool-calling loop exceeded "
                f"{MAX_TOOL_ROUNDS} rounds"
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            f"AI provider error: {error}"
        )

        raise HTTPException(
            status_code=502,
            detail="AI provider request failed",
        )

    finally:
        await close_user_tools(
            connections
        )


def _resolve_tool_id(
    tools,
    model_tool_name: str,
) -> str:
    for tool in tools:
        function = tool.model_definition.get(
            "function",
            {},
        )

        if function.get("name") == model_tool_name:
            return tool.id

    raise ValueError(
        f"Model requested unknown tool: "
        f"{model_tool_name}"
    )


@app.post("/api/title")
async def generate_title(
    request: TitleRequest,
):
    try:
        provider = get_provider(
            request.provider
        )

        prompt = f"""
Generate a short, meaningful title for a chat based on the user's first message.

Requirements:
- 3 to 6 words
- Clear and specific to the user's topic
- No quotation marks
- No markdown
- No explanation
- Return only the title

User message:
{request.message}
"""

        title = await provider.generate(
            message=prompt,
            model=request.model,
            api_key=request.api_key,
        )

        return {
            "title": title.strip(),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        print(
            f"Title generation error: {error}"
        )

        raise HTTPException(
            status_code=502,
            detail="AI title generation failed",
        )
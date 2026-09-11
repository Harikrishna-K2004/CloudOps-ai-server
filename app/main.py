from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.providers import get_provider, get_provider_metadata


app = FastAPI(title="CloudOps AI Server")


class ChatRequest(BaseModel):
    userId: str
    message: str
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    tools: list[str] | None = None

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
    authorization: str | None = Header(default=None),
):
    try:
        api_key = None

        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:]

        provider = get_provider(provider_name)

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
        print(f"Model discovery error: {error}")

        return {
            "error": "Failed to fetch models",
        }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        provider = get_provider(request.provider)

        response = await provider.generate(
            message=request.message,
            model=request.model,
            api_key=request.api_key,
        )

        return {
            "response": response,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        print(f"AI provider error: {error}")

        raise HTTPException(
            status_code=502,
            detail="AI provider request failed",
        )
        

@app.post("/api/title")
async def generate_title(request: TitleRequest):
    try:
        provider = get_provider(request.provider)

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
        print(f"Title generation error: {error}")

        raise HTTPException(
            status_code=502,
            detail="AI title generation failed",
        )
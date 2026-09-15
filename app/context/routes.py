from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.context.embeddings import generate_embedding


router = APIRouter(
    prefix="/api/context",
    tags=["context"],
)


class EmbeddingRequest(BaseModel):
    text: str
    api_key: str


@router.post("/embed")
async def create_embedding(
    request: EmbeddingRequest,
):
    try:
        embedding = await generate_embedding(
            text=request.text,
            api_key=request.api_key,
        )

        return {
            "embedding": embedding,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        print(
            f"Context embedding error: {error}"
        )

        raise HTTPException(
            status_code=502,
            detail="Failed to generate embedding",
        )
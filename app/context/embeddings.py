from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


async def generate_embedding(
    text: str,
    api_key: str,
) -> list[float]:
    if not text.strip():
        raise ValueError(
            "Cannot generate an embedding for empty text"
        )

    if not api_key:
        raise ValueError(
            "Gemini API key is required for embeddings"
        )

    client = genai.Client(api_key=api_key)

    response = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    if not response.embeddings:
        raise ValueError(
            "Gemini returned no embedding"
        )

    embedding = response.embeddings[0].values

    if not embedding:
        raise ValueError(
            "Gemini returned an empty embedding"
        )

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{len(embedding)}"
        )

    return list(embedding)
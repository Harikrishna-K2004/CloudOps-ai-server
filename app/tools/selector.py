
import json
from typing import Any

from app.providers.base import AIProvider


def _parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()

    # Remove Markdown code fences if the model returns them.
    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Integration selector returned invalid JSON: {content}"
        ) from error

    if not isinstance(result, dict):
        raise ValueError(
            "Integration selector response must be a JSON object."
        )

    return result


async def select_integrations(
    provider: AIProvider,
    message: str,
    integrations: list[dict[str, str]],
    model: str | None,
    api_key: str | None,
) -> list[str]:

    if not integrations:
        return []

    catalog = json.dumps(
        integrations,
        ensure_ascii=False,
    )

    prompt = f"""
You are the integration-selection component of CloudOps.

Your job is to decide which connected integrations are
relevant to the user's request.

Available connected integrations:

{catalog}

User request:

{message}

Rules:
- Select only integrations that are genuinely relevant.
- You may select multiple integrations when required.
- Do not invent integrations.
- Do not select an integration merely because it might be useful.
- Return ONLY valid JSON.
- Do not use Markdown or code fences.
- Return exactly this format:

{{"integrations":["provider_id_1","provider_id_2"]}}

If no integration is needed, return:

{{"integrations":[]}}
"""

    response = await provider.chat(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        api_key=api_key,
        tools=None,
    )

    result = _parse_json_response(response.content)

    selected = result.get("integrations")

    if not isinstance(selected, list):
        raise ValueError(
            "Integration selector returned an invalid integrations list"
        )

    valid_provider_ids = {
        integration["provider_id"]
        for integration in integrations
    }

    selected_provider_ids: list[str] = []

    for provider_id in selected:
        if not isinstance(provider_id, str):
            continue

        if provider_id not in valid_provider_ids:
            continue

        if provider_id not in selected_provider_ids:
            selected_provider_ids.append(provider_id)

    return selected_provider_ids

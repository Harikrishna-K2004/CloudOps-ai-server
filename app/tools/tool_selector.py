import json
from typing import Any

from app.providers.base import AIProvider
from app.tools.schema import ToolDefinition


def _parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()

    # Handle Markdown code fences such as:
    # ```json
    # {"tools": [...]}
    # ```
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
            f"Tool selector returned invalid JSON: {content}"
        ) from error

    if not isinstance(result, dict):
        raise ValueError("Tool selector response must be a JSON object.")

    return result


async def select_tools(
    provider: AIProvider,
    message: str,
    tools: list[ToolDefinition],
    model: str,
    api_key: str,
) -> list[str]:
    if not tools:
        return []

    catalog = []

    for tool in tools:
        function = tool.model_definition.get("function", {})

        catalog.append(
            {
                "tool_id": tool.id,
                "name": function.get("name", tool.id),
                "description": function.get("description", ""),
            }
        )

    prompt = f"""
You are the tool-selection component of CloudOps.

Your job is to decide which tools are genuinely needed to answer the user's request.

Available tools:
{json.dumps(catalog, ensure_ascii=False)}

User request:
{message}

Rules:
- Select only tools that are genuinely relevant.
- You may select multiple tools if necessary.
- Do not invent tool IDs.
- Do not select tools merely because they might be useful.
- Return ONLY valid JSON.
- Do not include Markdown or code fences.
- Exact format:
{{"tools":["tool_id_1","tool_id_2"]}}
- If no tool is needed:
{{"tools":[]}}
"""

    response = await provider.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        api_key=api_key,
        tools=None,
    )

    result = _parse_json_response(response.content)

    selected = result.get("tools")

    if not isinstance(selected, list):
        raise ValueError(
            'Tool selector response must contain a "tools" array.'
        )

    available_ids = {tool.id for tool in tools}

    return [
        tool_id
        for tool_id in selected
        if isinstance(tool_id, str) and tool_id in available_ids
    ]
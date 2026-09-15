from typing import Any

from .schema import ToolDefinition


_tools: dict[str, ToolDefinition] = {}


def register_tool(tool: ToolDefinition) -> None:
    if tool.id in _tools:
        raise ValueError(
            f"Tool already registered: {tool.id}"
        )

    _tools[tool.id] = tool


def unregister_tool(tool_id: str) -> None:
    _tools.pop(tool_id, None)


def get_tool(
    tool_id: str,
) -> ToolDefinition | None:
    return _tools.get(tool_id)


def get_all_tools() -> list[ToolDefinition]:
    return list(_tools.values())


def get_model_tools(
    tools: list[ToolDefinition] | None = None,
) -> list[dict[str, Any]]:
    selected_tools = (
        tools if tools is not None else get_all_tools()
    )

    return [
        tool.model_definition
        for tool in selected_tools
    ]


async def execute_tool(
    tool_id: str,
    arguments: dict[str, Any],
    user_id: str,
    tools: list[ToolDefinition] | None = None,
) -> Any:
    selected_tools = (
        tools if tools is not None else get_all_tools()
    )

    tool = next(
        (
            item
            for item in selected_tools
            if item.id == tool_id
        ),
        None,
    )

    if tool is None:
        raise ValueError(
            f"Tool not found: {tool_id}"
        )

    return await tool.execute(
        arguments,
        user_id,
    )
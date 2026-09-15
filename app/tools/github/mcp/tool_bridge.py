from typing import Any

from app.tools.schema import ToolDefinition
from .adapter import GitHubMCPAdapter


def create_github_tool(
    adapter: GitHubMCPAdapter,
    mcp_tool: Any,
) -> ToolDefinition:
    """
    Wrap an MCP tool without modifying its definition.
    """

    tool_name = mcp_tool.name

    if not tool_name:
        raise ValueError("MCP tool is missing a name")

    model_definition = {
        "type": "function",
        "function": {
            "name": f"github_mcp_{tool_name}",
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.input_schema,
        },
    }

    async def execute(
        arguments: dict[str, Any],
        user_id: str,
    ) -> Any:
        return await adapter.execute(
            tool_name,
            arguments,
        )

    return ToolDefinition(
        id=f"github.mcp.{tool_name}",
        provider_id="github",
        source="mcp",
        model_definition=model_definition,
        execute=execute,
        requires_connection=True,
        raw_definition=mcp_tool.model_dump(
            by_alias=True,
            exclude_none=False,
        ),
    )
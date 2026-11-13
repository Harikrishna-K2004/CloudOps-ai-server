
from typing import Any

from app.tools.schema import ToolDefinition
from .adapter import JenkinsMCPAdapter


def create_jenkins_tool(
    adapter: JenkinsMCPAdapter,
    mcp_tool: Any,
) -> ToolDefinition:
    """
    Convert a Jenkins MCP tool into a CloudOps tool.
    """

    tool_name = mcp_tool.name

    if not tool_name:
        raise ValueError(
            "Jenkins MCP tool is missing a name"
        )

    model_definition = {
        "type": "function",
        "function": {
            "name": f"jenkins_mcp_{tool_name}",
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
        id=f"jenkins.mcp.{tool_name}",
        provider_id="jenkins",
        source="mcp",
        model_definition=model_definition,
        execute=execute,
        requires_connection=True,
        raw_definition=mcp_tool.model_dump(
            by_alias=True,
            exclude_none=False,
        ),
    )

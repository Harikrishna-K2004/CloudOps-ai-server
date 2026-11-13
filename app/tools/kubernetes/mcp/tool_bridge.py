
from typing import Any

from app.tools.schema import ToolDefinition
from .adapter import KubernetesMCPAdapter


def create_kubernetes_tool(
    adapter: KubernetesMCPAdapter,
    mcp_tool: Any,
) -> ToolDefinition:
    tool_name = mcp_tool.name

    if not tool_name:
        raise ValueError(
            "Kubernetes MCP tool is missing a name"
        )

    input_schema = mcp_tool.input_schema or {}

    properties = input_schema.get("properties", {})

    if "namespace" in properties:
        adapter.set_namespace_aware_tools(
            adapter._namespace_aware_tools | {tool_name}
        )

    model_definition = {
        "type": "function",
        "function": {
            "name": f"kubernetes_mcp_{tool_name}",
            "description": mcp_tool.description or "",
            "parameters": input_schema,
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
        id=f"kubernetes.mcp.{tool_name}",
        provider_id="kubernetes",
        source="mcp",
        model_definition=model_definition,
        execute=execute,
        requires_connection=True,
        raw_definition=mcp_tool.model_dump(
            by_alias=True,
            exclude_none=False,
        ),
    )

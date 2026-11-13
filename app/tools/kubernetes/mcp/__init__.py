
from typing import Any

from .adapter import KubernetesMCPAdapter
from .tool_bridge import create_kubernetes_tool


async def discover_kubernetes_mcp_tools(
    credentials: dict[str, Any],
):
    adapter = KubernetesMCPAdapter(credentials)

    try:
        await adapter.connect()

        mcp_tools = await adapter.list_tools()

        tools = [
            create_kubernetes_tool(
                adapter,
                mcp_tool,
            )
            for mcp_tool in mcp_tools
        ]

        return tools, adapter

    except Exception:
        await adapter.disconnect()
        raise

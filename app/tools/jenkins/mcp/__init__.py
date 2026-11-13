
from typing import Any

from .adapter import JenkinsMCPAdapter
from .tool_bridge import create_jenkins_tool


async def discover_jenkins_mcp_tools(
    credentials: dict[str, Any],
):
    adapter = JenkinsMCPAdapter(credentials)

    try:
        await adapter.connect()

        mcp_tools = await adapter.list_tools()

        tools = [
            create_jenkins_tool(
                adapter,
                mcp_tool,
            )
            for mcp_tool in mcp_tools
        ]

        return tools, adapter

    except Exception:
        await adapter.disconnect()
        raise

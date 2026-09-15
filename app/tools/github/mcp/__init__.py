from typing import Any

from .adapter import GitHubMCPAdapter
from .tool_bridge import create_github_tool


async def discover_github_mcp_tools(
    credentials: dict[str, Any],
):
    adapter = GitHubMCPAdapter(credentials)

    await adapter.connect()

    mcp_tools = await adapter.list_tools()

    tools = [
        create_github_tool(
            adapter,
            mcp_tool,
        )
        for mcp_tool in mcp_tools
    ]

    return tools, adapter
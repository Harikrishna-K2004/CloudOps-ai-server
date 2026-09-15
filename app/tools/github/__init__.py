from typing import Any

from .mcp import discover_github_mcp_tools


async def discover_tools(
    credentials: dict[str, Any],
):
    return await discover_github_mcp_tools(
        credentials
    )
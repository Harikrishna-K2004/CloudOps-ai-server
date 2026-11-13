
from typing import Any

from .mcp import discover_jenkins_mcp_tools


async def discover_tools(
    credentials: dict[str, Any],
):
    return await discover_jenkins_mcp_tools(
        credentials
    )


from typing import Any

from .mcp import discover_kubernetes_mcp_tools


async def discover_tools(
    credentials: dict[str, Any],
):
    return await discover_kubernetes_mcp_tools(
        credentials
    )

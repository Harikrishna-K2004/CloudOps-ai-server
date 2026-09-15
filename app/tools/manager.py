from typing import Any

from .integration_registry import (
    get_all_integrations,
    get_discoverer,
    get_integration_catalog,
)
from .schema import ToolDefinition


def get_user_integration_catalog(
    tool_credentials: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Return only integrations that are enabled and connected
    for the current user.

    No MCP connections are opened and no tools are discovered.
    """

    connected_provider_ids = set(
        tool_credentials.keys()
    )

    return [
        integration
        for integration in get_integration_catalog()
        if integration["provider_id"]
        in connected_provider_ids
    ]

async def build_selected_integration_tools(
    selected_provider_ids: list[str],
    tool_credentials: dict[str, dict[str, Any]],
) -> tuple[list[ToolDefinition], list[Any]]:
    """
    Stage 2:
    Discover tools only for the integrations selected by Stage 1.
    """

    tools: list[ToolDefinition] = []
    connections: list[Any] = []

    selected = set(selected_provider_ids)

    for integration in get_all_integrations():
        if not integration.get("is_enabled", True):
            continue

        provider_id = integration["provider_id"]

        if provider_id not in selected:
            continue

        credentials = tool_credentials.get(provider_id)

        if credentials is None:
            continue

        discoverer = get_discoverer(provider_id)

        if discoverer is None:
            continue

        discovered_tools, integration_connection = (
            await discoverer(credentials)
        )

        tools.extend(discovered_tools)

        if integration_connection is not None:
            connections.append(integration_connection)

    return tools, connections


async def build_user_tools(
    user_id: str,
    tool_credentials: dict[str, dict[str, Any]],
) -> tuple[list[ToolDefinition], list[Any]]:
    """
    Build the complete tool set available to one user.

    This is used only after integration/tool selection.
    """

    tools: list[ToolDefinition] = []
    connections: list[Any] = []

    for integration in get_all_integrations():
        if not integration.get("is_enabled", True):
            continue

        provider_id = integration["provider_id"]

        credentials = tool_credentials.get(provider_id)

        if credentials is None:
            continue

        discoverer = get_discoverer(provider_id)

        if discoverer is None:
            continue

        discovered_tools, integration_connection = (
            await discoverer(credentials)
        )

        tools.extend(discovered_tools)

        if integration_connection is not None:
            connections.append(integration_connection)

    return tools, connections

def get_selected_tools(
    tools: list[ToolDefinition],
    selected_tool_ids: list[str],
) -> list[ToolDefinition]:
    """
    Stage 3:
    Return only the complete tool definitions selected by Stage 2.
    """

    selected = set(selected_tool_ids)

    return [
        tool
        for tool in tools
        if tool.id in selected
    ]


async def close_user_tools(
    connections: list[Any],
) -> None:
    """
    Close all request-scoped integration connections.
    """

    for connection in reversed(connections):
        disconnect = getattr(
            connection,
            "disconnect",
            None,
        )

        if disconnect is not None:
            await disconnect()
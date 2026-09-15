from typing import Any, Awaitable, Callable


IntegrationDiscoverer = Callable[
    [dict[str, Any]],
    Awaitable[tuple[list[Any], list[Any]]],
]

_integrations: dict[str, dict[str, Any]] = {}
_discoverers: dict[str, IntegrationDiscoverer] = {}


def register_integration(
    integration: dict[str, Any],
    discoverer: IntegrationDiscoverer | None = None,
) -> None:
    provider_id = integration.get("provider_id")

    if not provider_id:
        raise ValueError(
            "Integration must have a provider_id"
        )

    if provider_id in _integrations:
        raise ValueError(
            f"Integration already registered: {provider_id}"
        )

    _integrations[provider_id] = integration

    if discoverer is not None:
        _discoverers[provider_id] = discoverer


def get_integration(
    provider_id: str,
) -> dict[str, Any] | None:
    return _integrations.get(provider_id)


def get_all_integrations() -> list[dict[str, Any]]:
    return list(_integrations.values())


def get_discoverer(
    provider_id: str,
) -> IntegrationDiscoverer | None:
    return _discoverers.get(provider_id)


def get_integration_catalog() -> list[dict[str, str]]:
    """
    Return the lightweight integration catalog exposed to the AI.

    Only identification and descriptive information is returned.
    Credentials, connection schemas, discoverers, and tool schemas
    are never exposed here.
    """

    catalog: list[dict[str, str]] = []

    for integration in get_all_integrations():
        if not integration.get("is_enabled", True):
            continue

        catalog.append(
            {
                "provider_id": integration["provider_id"],
                "name": integration.get(
                    "name",
                    integration["provider_id"],
                ),
                "description": integration.get(
                    "description",
                    "",
                ),
            }
        )

    return catalog
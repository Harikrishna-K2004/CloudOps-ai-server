import importlib
import pkgutil

from .integration_registry import register_integration


def discover_integrations() -> None:
    tools_package = importlib.import_module("app.tools")

    for module_info in pkgutil.iter_modules(
        tools_package.__path__
    ):
        if (
            not module_info.ispkg
            or module_info.name.startswith("_")
        ):
            continue

        provider_id = module_info.name

        integration_module_name = (
            f"app.tools.{provider_id}.integration"
        )

        try:
            integration_module = importlib.import_module(
                integration_module_name
            )
        except ModuleNotFoundError as error:
            if error.name == integration_module_name:
                continue
            raise

        integration = None

        for attribute_name in dir(
            integration_module
        ):
            if not attribute_name.endswith(
                "_INTEGRATION"
            ):
                continue

            candidate = getattr(
                integration_module,
                attribute_name,
            )

            if isinstance(candidate, dict):
                integration = candidate
                break

        if integration is None:
            continue

        package = importlib.import_module(
            f"app.tools.{provider_id}"
        )

        discoverer = getattr(
            package,
            "discover_tools",
            None,
        )

        register_integration(
            integration,
            discoverer,
        )
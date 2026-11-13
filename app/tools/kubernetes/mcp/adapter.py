
from typing import Any
from urllib.parse import urlparse

from app.tools.mcp.client import MCPClient


class KubernetesMCPAdapter:
    """
    Adapter for one dedicated Kubernetes MCP server
    connected to one Kubernetes cluster.
    """

    def __init__(self, credentials: dict[str, Any]):
        self.credentials = credentials
        self.client: MCPClient | None = None

        self.namespace = (
            credentials.get("namespace") or ""
        ).strip() or None

    def _get_mcp_url(self) -> str:
        base_url = (
            self.credentials.get("mcp_server_url") or ""
        ).strip().rstrip("/")

        if not base_url:
            raise ValueError(
                "Kubernetes MCP Server URL is required"
            )

        parsed = urlparse(base_url)

        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "Invalid Kubernetes MCP Server URL"
            )

        if parsed.path.rstrip("/").endswith("/mcp"):
            return base_url

        return f"{base_url}/mcp"

    async def connect(self) -> None:
        if self.client is not None:
            return

        token = self.credentials.get(
            "service_account_token"
        )

        if not token:
            raise ValueError(
                "Kubernetes ServiceAccount token is required"
            )

        self.client = MCPClient(
            url=self._get_mcp_url(),
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        try:
            await self.client.connect()
        except Exception:
            await self.client.disconnect()
            self.client = None
            raise

    async def disconnect(self) -> None:
        if self.client is None:
            return

        await self.client.disconnect()
        self.client = None

    async def list_tools(self) -> list[Any]:
        if self.client is None:
            raise RuntimeError(
                "Kubernetes MCP adapter is not connected"
            )

        return await self.client.list_tools()

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if self.client is None:
            raise RuntimeError(
                "Kubernetes MCP adapter is not connected"
            )

        arguments = dict(arguments)

        # Apply the user's default namespace only when:
        # 1. A namespace is configured.
        # 2. The MCP tool accepts a namespace argument.
        # 3. The model did not provide one.
        #
        # Cluster-scoped tools remain unaffected.
        if (
            self.namespace
            and "namespace" not in arguments
            and tool_name in self._namespace_aware_tools
        ):
            arguments["namespace"] = self.namespace

        return await self.client.call_tool(
            tool_name,
            arguments,
        )

    @property
    def _namespace_aware_tools(self) -> set[str]:
        return getattr(
            self,
            "_namespace_tool_names",
            set(),
        )

    def set_namespace_aware_tools(
        self,
        tool_names: set[str],
    ) -> None:
        self._namespace_tool_names = tool_names

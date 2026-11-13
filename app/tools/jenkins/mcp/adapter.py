
import base64
from typing import Any
from urllib.parse import urlparse

from app.tools.mcp.client import MCPClient


class JenkinsMCPAdapter:
    """
    Adapter for the official Jenkins MCP Server plugin.
    """

    def __init__(
        self,
        credentials: dict[str, Any],
    ):
        self.credentials = credentials
        self.client: MCPClient | None = None

    def _get_connection_details(self):
        jenkins_url = self.credentials.get(
            "jenkins_url"
        )

        username = self.credentials.get(
            "username"
        )

        api_token = self.credentials.get(
            "api_token"
        )

        if not all([jenkins_url, username, api_token]):
            raise ValueError(
                "Jenkins URL, username, and API token are required"
            )

        jenkins_url = str(jenkins_url).strip().rstrip("/")

        parsed_url = urlparse(jenkins_url)

        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(
                "Jenkins URL must use HTTP or HTTPS"
            )

        if not parsed_url.netloc:
            raise ValueError(
                "Invalid Jenkins URL"
            )

        # Accept either the Jenkins base URL or the
        # complete Streamable HTTP MCP endpoint.
        if parsed_url.path.rstrip("/").endswith(
            "/mcp-server/mcp"
        ):
            mcp_url = jenkins_url
        else:
            mcp_url = (
                f"{jenkins_url}/mcp-server/mcp"
            )

        raw_credentials = (
            f"{username}:{api_token}"
        ).encode("utf-8")

        encoded_credentials = base64.b64encode(
            raw_credentials
        ).decode("ascii")

        return mcp_url, {
            "Authorization": (
                f"Basic {encoded_credentials}"
            ),
        }

    async def connect(self) -> None:
        if self.client is not None:
            return

        mcp_url, headers = (
            self._get_connection_details()
        )

        self.client = MCPClient(
            url=mcp_url,
            headers=headers,
        )

        try:
            await self.client.connect()
        except Exception:
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
                "Jenkins MCP adapter is not connected"
            )

        return await self.client.list_tools()

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if self.client is None:
            raise RuntimeError(
                "Jenkins MCP adapter is not connected"
            )

        return await self.client.call_tool(
            tool_name,
            arguments,
        )

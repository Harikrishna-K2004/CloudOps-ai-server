from typing import Any

from mcp import Client
from app.tools.mcp.client import MCPClient


GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"


class GitHubMCPAdapter:
    """
    Adapter for GitHub's hosted remote MCP server.
    """

    def __init__(self, credentials: dict[str, Any]):
        self.credentials = credentials
        self.client: MCPClient | None = None

    async def connect(self) -> None:
        if self.client is not None:
            return

        token = self.credentials.get("personal_access_token")

        if not token:
            raise ValueError(
                "GitHub personal access token is required"
            )

        self.client = MCPClient(
            url=GITHUB_MCP_URL,
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        await self.client.connect()

    async def disconnect(self) -> None:
        if self.client is None:
            return

        await self.client.disconnect()
        self.client = None

    async def list_tools(self) -> list[Any]:
        if self.client is None:
            raise RuntimeError(
                "GitHub MCP adapter is not connected"
            )

        return await self.client.list_tools()

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if self.client is None:
            raise RuntimeError(
                "GitHub MCP adapter is not connected"
            )

        return await self.client.call_tool(
            tool_name,
            arguments,
        )
from contextlib import AsyncExitStack
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


class MCPClient:
    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ):
        self.url = url
        self.headers = headers or {}
        self._exit_stack = AsyncExitStack()
        self._client: Client | None = None

    async def connect(self):
        if self._client is not None:
            return

        http_client = httpx2.AsyncClient(
            headers=self.headers,
            timeout=httpx2.Timeout(
                30.0,
                read=300.0,
            ),
        )

        await self._exit_stack.enter_async_context(
            http_client
        )

        transport = streamable_http_client(
            self.url,
            http_client=http_client,
        )

        client = Client(transport)

        self._client = await self._exit_stack.enter_async_context(
            client
        )

    async def disconnect(self):
        if self._client is None:
            return

        await self._exit_stack.aclose()

        self._exit_stack = AsyncExitStack()
        self._client = None

    async def list_tools(self):
        if self._client is None:
            raise RuntimeError(
                "MCP client is not connected"
            )

        result = await self._client.list_tools()

        return list(result.tools)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ):
        if self._client is None:
            raise RuntimeError(
                "MCP client is not connected"
            )

        return await self._client.call_tool(
            tool_name,
            arguments=arguments,
        )
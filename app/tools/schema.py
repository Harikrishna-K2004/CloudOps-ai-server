from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal


ToolSource = Literal["custom", "mcp"]


@dataclass
class ToolDefinition:
    """
    Common representation of every CloudOps tool.

    MCP and custom tools use exactly the same abstraction.

    `model_definition` is the complete definition that is exposed
    to the AI model.

    `raw_definition` preserves the original provider definition
    when one exists, such as an MCP tool definition.
    """

    id: str
    provider_id: str
    source: ToolSource

    model_definition: dict[str, Any]

    execute: Callable[
        [dict[str, Any], str],
        Awaitable[Any],
    ]

    requires_connection: bool = True

    raw_definition: dict[str, Any] | None = None
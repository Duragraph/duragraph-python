"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    raw_response: Any = None


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the provider.

        Args:
            api_key: API key for the provider.
            base_url: Optional custom base URL.
            **kwargs: Additional provider-specific options.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.options = kwargs

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            messages: List of messages in the conversation.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            tools: List of tool definitions for function calling.
            **kwargs: Additional provider-specific options.

        Returns:
            LLMResponse with the completion.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion.

        Args:
            messages: List of messages in the conversation.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            tools: List of tool definitions for function calling.
            **kwargs: Additional provider-specific options.

        Yields:
            StreamChunk objects as they arrive.
        """
        ...

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming."""
        return True

    def supports_tools(self) -> bool:
        """Check if this provider supports tool/function calling."""
        return True

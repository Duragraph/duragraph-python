"""OpenAI LLM provider."""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from duragraph.llm.base import LLMProvider, LLMResponse, Message, StreamChunk, ToolCall

try:
    from openai import AsyncOpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    AsyncOpenAI = None  # type: ignore


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            base_url: Optional custom base URL (for Azure or proxies).
            organization: Optional organization ID.
            **kwargs: Additional options.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install duragraph-python[openai]"
            )

        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            organization=organization,
        )

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
        """Generate a completion using OpenAI."""
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)

        # Build request
        request: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
        }

        if max_tokens:
            request["max_tokens"] = max_tokens

        if tools:
            request["tools"] = self._convert_tools(tools)

        # Add any extra kwargs
        request.update(kwargs)

        # Make request
        response = await self.client.chat.completions.create(**request)

        # Parse response
        choice = response.choices[0]
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason or "stop",
            raw_response=response,
        )

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
        """Stream a completion using OpenAI."""
        openai_messages = self._convert_messages(messages)

        request: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if max_tokens:
            request["max_tokens"] = max_tokens

        if tools:
            request["tools"] = self._convert_tools(tools)

        request.update(kwargs)

        stream = await self.client.chat.completions.create(**request)

        # Accumulate tool calls across chunks
        current_tool_calls: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                # Final chunk with usage
                if chunk.usage:
                    yield StreamChunk(
                        usage={
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    )
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            content = delta.content or ""
            tool_calls = []

            # Handle tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc_delta.id:
                        current_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            current_tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            current_tool_calls[idx]["arguments"] += tc_delta.function.arguments

            # Emit complete tool calls on finish
            if finish_reason == "tool_calls":
                for tc_data in current_tool_calls.values():
                    try:
                        args = json.loads(tc_data["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            arguments=args,
                        )
                    )

            yield StreamChunk(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        result = []
        for msg in messages:
            m: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            result.append(m)
        return result

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool definitions to OpenAI format."""
        result = []
        for tool in tools:
            if "type" not in tool:
                # Assume it's a function definition
                result.append({"type": "function", "function": tool})
            else:
                result.append(tool)
        return result

"""Anthropic LLM provider."""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from duragraph.llm.base import LLMProvider, LLMResponse, Message, StreamChunk, ToolCall

try:
    from anthropic import AsyncAnthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    AsyncAnthropic = None  # type: ignore


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            base_url: Optional custom base URL.
            **kwargs: Additional options.
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic package not installed. "
                "Install with: pip install duragraph-python[anthropic]"
            )

        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

        self.client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url,
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
        """Generate a completion using Anthropic."""
        # Extract system message
        system_message = None
        conv_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conv_messages.append(msg)

        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages(conv_messages)

        # Build request
        request: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }

        if system_message:
            request["system"] = system_message

        if tools:
            request["tools"] = self._convert_tools(tools)

        request.update(kwargs)

        # Make request
        response = await self.client.messages.create(**request)

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            content=content,
            model=response.model,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason or "end_turn",
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
        """Stream a completion using Anthropic."""
        # Extract system message
        system_message = None
        conv_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conv_messages.append(msg)

        anthropic_messages = self._convert_messages(conv_messages)

        request: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }

        if system_message:
            request["system"] = system_message

        if tools:
            request["tools"] = self._convert_tools(tools)

        request.update(kwargs)

        # Track tool use blocks
        current_tool: dict[str, Any] | None = None
        input_tokens = 0
        output_tokens = 0

        async with self.client.messages.stream(**request) as stream:
            async for event in stream:
                if event.type == "message_start":
                    if event.message.usage:
                        input_tokens = event.message.usage.input_tokens

                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "arguments": "",
                        }

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamChunk(content=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        if current_tool:
                            current_tool["arguments"] += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool:
                        try:
                            args = json.loads(current_tool["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                        yield StreamChunk(
                            tool_calls=[
                                ToolCall(
                                    id=current_tool["id"],
                                    name=current_tool["name"],
                                    arguments=args,
                                )
                            ]
                        )
                        current_tool = None

                elif event.type == "message_delta":
                    if event.usage:
                        output_tokens = event.usage.output_tokens
                    yield StreamChunk(
                        finish_reason=event.delta.stop_reason,
                        usage={
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to Anthropic format."""
        result = []
        for msg in messages:
            role = msg.role
            if role == "tool":
                role = "user"
                # Tool results need special format
                result.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
            elif role == "assistant" and msg.tool_calls:
                # Assistant with tool calls
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                        }
                    )
                result.append({"role": role, "content": content})
            else:
                # Map 'user' to 'user' and 'assistant' to 'assistant'
                if role == "human":
                    role = "user"
                result.append({"role": role, "content": msg.content})
        return result

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool definitions to Anthropic format."""
        result = []
        for tool in tools:
            # Handle OpenAI-style tool definitions
            if "type" in tool and tool["type"] == "function":
                func = tool["function"]
                result.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object"}),
                    }
                )
            elif "name" in tool:
                # Already in Anthropic format
                result.append(tool)
            else:
                # Assume it's a function definition directly
                result.append(
                    {
                        "name": tool.get("name", "unknown"),
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters", {"type": "object"}),
                    }
                )
        return result

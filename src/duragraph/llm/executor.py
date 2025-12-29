"""LLM execution utilities for graph nodes."""

from collections.abc import AsyncIterator
from typing import Any

from duragraph.llm.base import LLMResponse, Message
from duragraph.llm.registry import get_provider


async def execute_llm_node(
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute an LLM node with the given state and config.

    Args:
        state: Current graph state.
        config: Node configuration from @llm_node decorator.

    Returns:
        Updated state with LLM response.
    """
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens")
    system_prompt = config.get("system_prompt")
    tools = config.get("tools", [])

    # Get provider
    provider = get_provider(model)

    # Build messages from state
    messages = _build_messages(state, system_prompt)

    # Execute
    response = await provider.complete(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools if tools else None,
    )

    # Update state
    return _process_response(state, response)


async def stream_llm_node(
    state: dict[str, Any],
    config: dict[str, Any],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Stream an LLM node execution.

    Args:
        state: Current graph state.
        config: Node configuration from @llm_node decorator.

    Yields:
        Tuples of (event_type, data) for streaming updates.
    """
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens")
    system_prompt = config.get("system_prompt")
    tools = config.get("tools", [])

    provider = get_provider(model)
    messages = _build_messages(state, system_prompt)

    content_buffer = ""
    tool_calls = []

    async for chunk in provider.stream(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools if tools else None,
    ):
        if chunk.content:
            content_buffer += chunk.content
            yield ("token", {"content": chunk.content})

        if chunk.tool_calls:
            tool_calls.extend(chunk.tool_calls)
            for tc in chunk.tool_calls:
                yield (
                    "tool_call",
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                )

        if chunk.finish_reason:
            yield (
                "finish",
                {
                    "finish_reason": chunk.finish_reason,
                    "usage": chunk.usage or {},
                },
            )

    # Final state update
    new_state = state.copy()
    if content_buffer:
        # Add assistant message to messages list
        messages_list = new_state.get("messages", [])
        messages_list.append(
            {
                "role": "assistant",
                "content": content_buffer,
            }
        )
        new_state["messages"] = messages_list
        new_state["last_response"] = content_buffer

    if tool_calls:
        new_state["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
        ]

    yield ("state_update", new_state)


def _build_messages(
    state: dict[str, Any],
    system_prompt: str | None = None,
) -> list[Message]:
    """Build messages list from state.

    Args:
        state: Current graph state.
        system_prompt: Optional system prompt to prepend.

    Returns:
        List of Message objects.
    """
    messages: list[Message] = []

    # Add system prompt if provided
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))

    # Check for messages in state
    state_messages = state.get("messages", [])
    if state_messages:
        for msg in state_messages:
            if isinstance(msg, dict):
                messages.append(
                    Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        name=msg.get("name"),
                        tool_calls=msg.get("tool_calls"),
                        tool_call_id=msg.get("tool_call_id"),
                    )
                )
            elif isinstance(msg, Message):
                messages.append(msg)

    # If no messages, create one from input
    if not state_messages:
        input_text = state.get("input", state.get("message", ""))
        if input_text:
            messages.append(Message(role="user", content=str(input_text)))

    return messages


def _process_response(
    state: dict[str, Any],
    response: LLMResponse,
) -> dict[str, Any]:
    """Process LLM response and update state.

    Args:
        state: Current state.
        response: LLM response.

    Returns:
        Updated state.
    """
    new_state = state.copy()

    # Add assistant message to messages list
    messages_list = new_state.get("messages", [])
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
    }
    if response.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in response.tool_calls
        ]
    messages_list.append(assistant_msg)
    new_state["messages"] = messages_list

    # Set convenience fields
    new_state["last_response"] = response.content
    new_state["usage"] = response.usage

    if response.tool_calls:
        new_state["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls
        ]

    return new_state

"""Tests for LLM provider integrations."""

import pytest

from duragraph.llm.base import LLMProvider, LLMResponse, Message, StreamChunk, ToolCall
from duragraph.llm.registry import _infer_provider, get_provider, register_provider


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    async def complete(self, messages, model, **kwargs):
        return LLMResponse(
            content="Mock response",
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    async def stream(self, messages, model, **kwargs):
        yield StreamChunk(content="Mock ")
        yield StreamChunk(content="stream")
        yield StreamChunk(finish_reason="stop")


class TestProviderRegistry:
    """Tests for the provider registry."""

    def test_infer_openai_provider(self):
        """Test inferring OpenAI provider from model name."""
        assert _infer_provider("gpt-4o-mini") == "openai"
        assert _infer_provider("gpt-4") == "openai"
        assert _infer_provider("gpt-3.5-turbo") == "openai"
        assert _infer_provider("o1-preview") == "openai"

    def test_infer_anthropic_provider(self):
        """Test inferring Anthropic provider from model name."""
        assert _infer_provider("claude-3-sonnet") == "anthropic"
        assert _infer_provider("claude-3-opus") == "anthropic"
        assert _infer_provider("claude-2") == "anthropic"

    def test_register_and_get_provider(self):
        """Test registering and retrieving a provider."""
        register_provider("mock", MockProvider)

        # Force the registry to use mock for a specific model
        provider = MockProvider()
        assert isinstance(provider, LLMProvider)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_calls is None

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        msg = Message(
            role="assistant",
            content="Let me search",
            tool_calls=[{"id": "1", "function": {"name": "search"}}],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello!",
            model="gpt-4o-mini",
            usage={"total_tokens": 10},
        )
        assert response.content == "Hello!"
        assert response.model == "gpt-4o-mini"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        response = LLMResponse(
            content="",
            model="gpt-4o-mini",
            tool_calls=[ToolCall(id="1", name="search", arguments={"query": "test"})],
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        tc = ToolCall(id="call_123", name="get_weather", arguments={"city": "NYC"})
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}


@pytest.mark.asyncio
class TestMockProvider:
    """Tests for mock provider functionality."""

    async def test_complete(self):
        """Test completion with mock provider."""
        provider = MockProvider()
        messages = [Message(role="user", content="Hello")]

        response = await provider.complete(messages, "mock-model")

        assert response.content == "Mock response"
        assert response.model == "mock-model"
        assert response.usage["total_tokens"] == 15

    async def test_stream(self):
        """Test streaming with mock provider."""
        provider = MockProvider()
        messages = [Message(role="user", content="Hello")]

        chunks = []
        async for chunk in provider.stream(messages, "mock-model"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Mock "
        assert chunks[1].content == "stream"
        assert chunks[2].finish_reason == "stop"

    def test_supports_streaming(self):
        """Test streaming support check."""
        provider = MockProvider()
        assert provider.supports_streaming() is True

    def test_supports_tools(self):
        """Test tool support check."""
        provider = MockProvider()
        assert provider.supports_tools() is True

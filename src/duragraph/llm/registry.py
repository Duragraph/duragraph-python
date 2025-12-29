"""LLM provider registry."""

from typing import Any

from duragraph.llm.base import LLMProvider

# Global provider registry
_providers: dict[str, type[LLMProvider]] = {}
_instances: dict[str, LLMProvider] = {}


def register_provider(name: str, provider_class: type[LLMProvider]) -> None:
    """Register an LLM provider.

    Args:
        name: Provider name (e.g., "openai", "anthropic").
        provider_class: The provider class to register.
    """
    _providers[name] = provider_class


def get_provider(
    model: str,
    *,
    api_key: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Get an LLM provider for a model.

    Args:
        model: Model identifier (e.g., "gpt-4o-mini", "claude-3-sonnet").
        api_key: Optional API key override.
        **kwargs: Additional provider options.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If no provider found for the model.
    """
    provider_name = _infer_provider(model)

    # Check cache
    cache_key = f"{provider_name}:{api_key or 'default'}"
    if cache_key in _instances and not kwargs:
        return _instances[cache_key]

    # Get provider class
    if provider_name not in _providers:
        # Try to import and register
        _auto_register_provider(provider_name)

    if provider_name not in _providers:
        raise ValueError(
            f"No provider found for model '{model}'. Available providers: {list(_providers.keys())}"
        )

    provider_class = _providers[provider_name]
    instance = provider_class(api_key=api_key, **kwargs)

    # Cache if no custom options
    if not kwargs:
        _instances[cache_key] = instance

    return instance


def _infer_provider(model: str) -> str:
    """Infer the provider from a model name.

    Args:
        model: Model identifier.

    Returns:
        Provider name.
    """
    model_lower = model.lower()

    # OpenAI models
    if any(
        m in model_lower
        for m in ["gpt-4", "gpt-3.5", "o1", "o3", "davinci", "curie", "babbage", "ada"]
    ):
        return "openai"

    # Anthropic models
    if any(m in model_lower for m in ["claude", "anthropic"]):
        return "anthropic"

    # Google models
    if any(m in model_lower for m in ["gemini", "palm", "bard"]):
        return "google"

    # Default to OpenAI
    return "openai"


def _auto_register_provider(name: str) -> None:
    """Attempt to auto-register a provider by importing it.

    Args:
        name: Provider name.
    """
    if name == "openai":
        try:
            from duragraph.llm.openai import OpenAIProvider

            register_provider("openai", OpenAIProvider)
        except ImportError:
            pass
    elif name == "anthropic":
        try:
            from duragraph.llm.anthropic import AnthropicProvider

            register_provider("anthropic", AnthropicProvider)
        except ImportError:
            pass

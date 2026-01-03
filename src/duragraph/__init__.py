"""DuraGraph Python SDK - Reliable AI Workflow Orchestration."""

from duragraph.edges import edge
from duragraph.graph import Graph
from duragraph.nodes import (
    entrypoint,
    human_node,
    llm_node,
    node,
    router_node,
    tool_node,
)
from duragraph.tools import tool, get_global_registry, resolve_tool_calls
from duragraph.types import AIMessage, HumanMessage, Message, State, ToolMessage

__version__ = "0.1.0"

__all__ = [
    # Graph
    "Graph",
    # Node decorators
    "node",
    "llm_node",
    "tool_node",
    "router_node",
    "human_node",
    "entrypoint",
    # Edge
    "edge",
    # Tools
    "tool",
    "get_global_registry",
    "resolve_tool_calls",
    # Types
    "State",
    "Message",
    "HumanMessage",
    "AIMessage",
    "ToolMessage",
]

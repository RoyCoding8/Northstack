"""Tests for the Anthropic write-side prompt cache.

Covers:
  - Nothing is marked unless the profile declares PROMPT_CACHING
  - Two breakpoints: the static prefix (ends at system) and the newest turn
  - A string message content is normalised to a block before it is marked
  - The static prefix is byte-identical across turns, so it actually caches
"""

from __future__ import annotations

import pytest

from northstack.adapters.providers.gateway import AnthropicAdapter, _mark_cache_breakpoint
from northstack.adapters.providers.wire import (
    MessageRole,
    ModelMessage,
    ModelRequest,
    ToolCall,
    ToolDefinition,
)
from northstack.config import Capability, ModelProfile, Protocol

_TOOLS = [
    ToolDefinition(name="read", description="read a file", parameters={"type": "object"}),
    ToolDefinition(name="write", description="write a file", parameters={"type": "object"}),
]


def _profile(*, caching: bool) -> ModelProfile:
    capabilities = {Capability.TOOL_USE}
    if caching:
        capabilities.add(Capability.PROMPT_CACHING)
    return ModelProfile(
        name="claude",
        protocol=Protocol.ANTHROPIC_MESSAGES,
        base_url="https://api.anthropic.com",
        model="claude-test",
        max_concurrency=4,
        capabilities=capabilities,
    )


def _request(turns: int = 0) -> ModelRequest:
    messages = [
        ModelMessage(role=MessageRole.SYSTEM, content="you are a worker"),
        ModelMessage(role=MessageRole.USER, content="Objective: build it"),
    ]
    for i in range(turns):
        messages.append(
            ModelMessage(
                role=MessageRole.ASSISTANT,
                content="calling",
                tool_calls=[ToolCall(id=f"c{i}", name="read", arguments={})],
            )
        )
        messages.append(
            ModelMessage(role=MessageRole.TOOL, content=f"contents {i}", tool_call_id=f"c{i}")
        )
    return ModelRequest(
        profile_name="claude", messages=messages, tools=list(_TOOLS), max_output_tokens=1000
    )


def _body(*, caching: bool, turns: int = 0) -> dict:
    return AnthropicAdapter()._build_body(_request(turns), _profile(caching=caching))


def _breakpoints(body: dict) -> int:
    count = sum(1 for t in body.get("tools", []) if "cache_control" in t)
    system = body.get("system")
    if isinstance(system, list):
        count += sum(1 for b in system if "cache_control" in b)
    for msg in body["messages"]:
        if isinstance(msg["content"], list):
            count += sum(1 for b in msg["content"] if "cache_control" in b)
    return count


class TestOptIn:
    def test_no_markers_without_the_capability(self):
        body = _body(caching=False, turns=2)
        assert _breakpoints(body) == 0
        assert isinstance(body["system"], str)

    def test_markers_appear_with_the_capability(self):
        assert _breakpoints(_body(caching=True, turns=2)) > 0


class TestBreakpoints:
    @pytest.mark.parametrize("turns", [0, 1, 5])
    def test_exactly_two_breakpoints(self, turns: int):
        """Anthropic allows four; two is the whole design -- one static, one
        rolling. More would evict each other for no gain.
        """
        assert _breakpoints(_body(caching=True, turns=turns)) == 2

    def test_static_breakpoint_sits_on_system_and_covers_the_tools(self):
        """A breakpoint caches the prefix up to and including its block, and
        ``tools`` precedes ``system`` in that prefix -- so marking a tool as
        well would only spend a breakpoint on ground already covered.
        """
        body = _body(caching=True, turns=1)
        assert all("cache_control" not in t for t in body["tools"])
        assert body["system"] == [
            {"type": "text", "text": "you are a worker", "cache_control": {"type": "ephemeral"}}
        ]

    def test_rolling_breakpoint_is_on_the_final_message(self):
        body = _body(caching=True, turns=2)
        assert "cache_control" in body["messages"][-1]["content"][-1]
        assert all(
            "cache_control" not in block
            for msg in body["messages"][:-1]
            if isinstance(msg["content"], list)
            for block in msg["content"]
        )

    def test_static_prefix_is_byte_identical_as_the_conversation_grows(self):
        """The point of the static breakpoint: it must not drift, or every
        turn pays a full cache write instead of a read.
        """
        first, later = _body(caching=True, turns=1), _body(caching=True, turns=6)
        assert first["tools"] == later["tools"]
        assert first["system"] == later["system"]


class TestMarkBreakpoint:
    def test_string_content_is_normalised_to_a_block(self):
        message = {"role": "user", "content": "hello"}
        _mark_cache_breakpoint(message)
        assert message["content"] == [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
        ]

    def test_only_the_last_block_is_marked(self):
        message = {"role": "user", "content": [{"type": "text", "text": a} for a in "abc"]}
        _mark_cache_breakpoint(message)
        assert [("cache_control" in b) for b in message["content"]] == [False, False, True]

    @pytest.mark.parametrize("content", ["", []])
    def test_empty_content_is_left_alone(self, content):
        message = {"role": "user", "content": content}
        _mark_cache_breakpoint(message)
        assert message["content"] == content

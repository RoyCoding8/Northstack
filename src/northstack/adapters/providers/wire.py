"""Provider-neutral wire models the gateway translates to/from vendor formats."""

from __future__ import annotations

import enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field


class MessageRole(str, enum.Enum):
    """Provider-neutral message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, enum.Enum):
    """Why the model stopped generating."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    ERROR = "error"


class ToolDefinition(BaseModel):
    """A tool the model may call, with a JSON-Schema parameter spec."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Unique tool name")
    description: str = Field(default="", description="Tool description for the model")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON-Schema for the tool's input parameters",
    )


class ToolCall(BaseModel):
    """A single tool call requested by the model."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Unique call identifier from the provider")
    name: str = Field(min_length=1, description="Tool name to invoke")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Parsed arguments")
    signature: str = Field(
        default="",
        description=(
            "Opaque provider token bound to this call, echoed back verbatim when the call "
            "appears in later history; empty for protocols that issue none"
        ),
    )


class ToolResultMessage(BaseModel):
    """A tool result fed back to the model after execution."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str = Field(min_length=1, description="Matches the ToolCall.id")
    content: str = Field(default="", description="Result content (text)")
    is_error: bool = Field(default=False, description="True if the tool execution failed")


class ImageContent(BaseModel):
    """An inline image attached to a message.

    Base64 rather than a URL: all three protocols accept inline bytes, only
    some accept a remote URL, and a URL would hand the provider a fetch the
    workspace's URL policy never saw.
    """

    model_config = ConfigDict(frozen=True)

    media_type: str = Field(
        pattern=r"^image/(png|jpeg|gif|webp)$",
        description="IANA media type; the intersection all three protocols accept",
    )
    data: str = Field(min_length=1, description="Base64-encoded image bytes, unprefixed")


class ModelMessage(BaseModel):
    """A single message in a provider-neutral conversation.

    ``role`` + ``content`` is the minimal pair.  Assistant messages that
    include tool calls populate ``tool_calls``; tool-role messages link back
    via ``tool_call_id`` and set ``is_error`` when the call failed.
    """

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str = Field(default="", description="Text content")
    images: list[ImageContent] = Field(
        default_factory=list,
        description="Inline images; requires Capability.VISION on the profile",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls (only for assistant messages)"
    )
    tool_call_id: str = Field(
        default="",
        description="Which tool call this result responds to (only for tool messages)",
    )
    is_error: bool = Field(
        default=False,
        description="True if the tool call this result responds to failed",
    )

    @classmethod
    def from_tool_result(cls, result: ToolResultMessage) -> ModelMessage:
        """Lift a executed tool result into the conversation without losing
        its error flag -- the two types carry the same three facts."""
        return cls(
            role=MessageRole.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
            is_error=result.is_error,
        )


class Usage(BaseModel):
    """Token usage for a single model call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TextDelta(BaseModel):
    """A fragment of assistant text."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ToolCallDelta(BaseModel):
    """An incremental tool-call fragment keyed by the provider's block index.

    ``arguments_fragment`` accumulates across deltas for one index; id/name
    arrive on the first delta of the block.  The gateway assembles complete
    ToolCalls once the stream closes.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call"] = "tool_call"
    index: int = Field(ge=0)
    id: str = ""
    name: str = ""
    arguments_fragment: str = ""
    signature: str = Field(
        default="", description="Opaque provider token for this call; see ToolCall.signature"
    )


class UsageDelta(BaseModel):
    """Usage observed so far; final totals arrive before the finish event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["usage"] = "usage"
    usage: Usage


class FinishDelta(BaseModel):
    """Terminal stream event; exactly one per stream."""

    model_config = ConfigDict(frozen=True)

    type: Literal["finish"] = "finish"
    finish_reason: FinishReason


StreamDelta = Annotated[
    Union[TextDelta, ToolCallDelta, UsageDelta, FinishDelta],
    Discriminator("type"),
]


class ModelRequest(BaseModel):
    """Provider-neutral model completion request.

    The gateway translates this into the wire format for the selected protocol
    (OpenAI-chat or Anthropic-messages).
    """

    model_config = ConfigDict(frozen=True)

    profile_name: str = Field(
        min_length=1, description="Which ModelProfile to use (resolves endpoint, auth, limits)"
    )
    messages: list[ModelMessage] = Field(min_length=1)
    system: str = Field(default="", description="System prompt (injected as system message)")
    tools: list[ToolDefinition] = Field(default_factory=list)
    output_json_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional JSON-Schema for structured output; only sent when profile supports it"
        ),
    )
    max_output_tokens: int | None = Field(
        default=None,
        description=(
            "Override per-request max output tokens; falls back to profile max_output_tokens"
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (provider must support; omit if not)",
    )
    seed: int | None = Field(
        default=None, description="Reproducibility seed (provider must support; omit if not)"
    )

    def get_max_tokens(self, profile_max: int) -> int:
        """Return the effective max_output_tokens."""
        return self.max_output_tokens if self.max_output_tokens is not None else profile_max


class ModelResponse(BaseModel):
    """Provider-normalized model completion response.

    The raw provider response is stored via ArtifactStore and referenced by
    ``response_artifact_id`` -- it is never embedded in event payloads.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(default="", description="Plain text output from the model")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls requested by the model"
    )
    finish_reason: FinishReason = Field(description="Why generation stopped")
    usage: Usage = Field(default_factory=Usage)
    provider: str = Field(description="Provider name (openai, anthropic, ...)")
    model: str = Field(description="Model identifier that served the response")
    response_artifact_id: str | None = Field(
        default=None,
        description="ArtifactRef digest for normalized response artifact; never raw provider body",
    )

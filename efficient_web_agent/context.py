"""Context-budget utilities and Pydantic AI history processing."""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart


def cap_text(text: str, limit: int, marker: str = "\n...[truncated]") -> tuple[str, bool]:
    """Return text within a hard character limit.

    Args:
        text: Source text to cap.
        limit: Maximum number of characters to return.
        marker: Suffix appended when text is truncated and space allows.

    Returns:
        Tuple of capped text and a flag indicating whether truncation happened.
    """

    if len(text) <= limit:
        return text, False
    if limit <= len(marker):
        return text[:limit], True
    return text[: limit - len(marker)] + marker, True


def _tool_call_ids(message: ModelMessage) -> set[str]:
    """Extract tool-call IDs from a Pydantic AI model response.

    Args:
        message: Pydantic AI model message to inspect.

    Returns:
        Set of tool call IDs present in `ToolCallPart` response parts.
    """

    if not isinstance(message, ModelResponse):
        return set()
    return {part.tool_call_id for part in message.parts if isinstance(part, ToolCallPart)}


def _tool_return_ids(message: ModelMessage) -> set[str]:
    """Extract tool-return IDs from a Pydantic AI model request.

    Args:
        message: Pydantic AI model message to inspect.

    Returns:
        Set of tool call IDs present in `ToolReturnPart` request parts.
    """

    if not isinstance(message, ModelRequest):
        return set()
    return {part.tool_call_id for part in message.parts if isinstance(part, ToolReturnPart)}


def _has_system_prompt(message: ModelMessage) -> bool:
    """Check whether a message contains a system prompt part.

    Args:
        message: Pydantic AI model message to inspect.

    Returns:
        True when the message is a ModelRequest containing a system-prompt part.
    """

    return isinstance(message, ModelRequest) and any(
        getattr(part, "part_kind", "") == "system-prompt" for part in message.parts
    )


async def trim_history(messages: list[ModelMessage], limit: int = 24) -> list[ModelMessage]:
    """Trim old messages without leaving orphaned tool return messages.

    Args:
        messages: Full Pydantic AI message history before a model request.
        limit: Maximum number of messages to retain.

    Returns:
        Trimmed message list. A leading system prompt is preserved, and a
        removed tool call also removes its immediately following tool return.
    """

    if len(messages) <= limit:
        return messages

    prefix: list[ModelMessage] = []
    body = list(messages)
    if body and _has_system_prompt(body[0]):
        prefix.append(body.pop(0))

    while len(prefix) + len(body) > limit and body:
        removed = body.pop(0)
        removed_calls = _tool_call_ids(removed)
        if removed_calls and body and _tool_return_ids(body[0]) & removed_calls:
            body.pop(0)
        elif isinstance(removed, ModelRequest) and _tool_return_ids(removed):
            continue

    return prefix + body


def history_processor(limit: int):
    """Create a Pydantic AI history processor that trims to a message limit.

    Args:
        limit: Maximum number of messages passed to `trim_history`.

    Returns:
        Async callable compatible with Pydantic AI `history_processors`.
    """

    async def processor(messages: list[ModelMessage], _ctx: Any = None) -> list[ModelMessage]:
        """Trim a history list for Pydantic AI.

        Args:
            messages: Message history supplied by Pydantic AI.
            _ctx: Optional processor context supplied by Pydantic AI and unused.

        Returns:
            Trimmed message history.
        """

        return await trim_history(messages, limit=limit)

    return processor

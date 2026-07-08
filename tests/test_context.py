from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from efficient_web_agent.context import cap_text, trim_history


def test_cap_text_marks_truncation() -> None:
    text, truncated = cap_text("abcdef", 5, marker="...")

    assert text == "ab..."
    assert truncated is True


async def test_trim_history_preserves_system_prompt() -> None:
    messages = [
        ModelRequest(parts=[SystemPromptPart(content="system")]),
        ModelRequest(parts=[UserPromptPart(content="old")]),
        ModelResponse(parts=[TextPart(content="old response")]),
        ModelRequest(parts=[UserPromptPart(content="new")]),
    ]

    trimmed = await trim_history(messages, limit=3)

    assert trimmed[0] is messages[0]
    assert len(trimmed) == 3
    assert trimmed[-1] is messages[-1]


async def test_trim_history_removes_tool_return_when_tool_call_is_trimmed() -> None:
    call = ToolCallPart(tool_name="search_page", args={"query": "x"}, tool_call_id="call-1")
    ret = ToolReturnPart(tool_name="search_page", content="result", tool_call_id="call-1")
    messages = [
        ModelRequest(parts=[SystemPromptPart(content="system")]),
        ModelResponse(parts=[call]),
        ModelRequest(parts=[ret]),
        ModelRequest(parts=[UserPromptPart(content="latest")]),
        ModelResponse(parts=[TextPart(content="answer")]),
    ]

    trimmed = await trim_history(messages, limit=3)

    serialized = repr(trimmed)
    assert "call-1" not in serialized
    assert trimmed[0] is messages[0]
    assert trimmed[-1] is messages[-1]

from types import SimpleNamespace

import pytest
from pydantic_ai import AgentRunResult
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_graph import End

from efficient_web_agent.agent import (
    _extract_source_urls,
    _fetch_max_model_len,
    _normalize_source_url,
    _print_context_usage,
    _run_agent_iter,
    _usage_to_dict,
    _validate_cited_sources_were_visited,
    create_agent,
    create_model,
)
from efficient_web_agent.settings import AgentSettings


def test_create_model_uses_openai_compatible_settings() -> None:
    settings = AgentSettings(model_name="local-model", base_url="http://0.0.0.0:8000/v1", api_key="api-key-not-set")

    model = create_model(settings)

    assert model.model_name == "local-model"


def test_create_agent_registers_context_safe_tools() -> None:
    settings = AgentSettings(model_name="local-model")

    agent = create_agent(settings)
    tool_names = set(agent._function_toolset.tools)  # noqa: SLF001 - verifies public behavior missing a public accessor.

    assert {
        "observe_page",
        "search_page",
        "extract_neighborhood",
        "click_element",
        "type_text",
        "websearch",
    } <= tool_names


def test_agent_tools_expose_argument_and_return_guidance() -> None:
    settings = AgentSettings(model_name="local-model")

    agent = create_agent(settings)

    for tool in agent._function_toolset.tools.values():  # noqa: SLF001 - verifies schema exposed to the model.
        assert "<returns>" in tool.description
        for property_schema in tool.function_schema.json_schema.get("properties", {}).values():
            assert property_schema.get("description")


def test_usage_to_dict_serializes_pydantic_ai_run_usage() -> None:
    usage = RunUsage(requests=1, tool_calls=2, input_tokens=3, output_tokens=4)

    serialized = _usage_to_dict(usage)

    assert serialized["requests"] == 1
    assert serialized["tool_calls"] == 2
    assert serialized["input_tokens"] == 3
    assert serialized["output_tokens"] == 4
    assert serialized["details"] == {}


def test_context_usage_prints_percentage_to_stderr(capsys) -> None:
    _print_context_usage(2, 2048, 8192)

    assert capsys.readouterr().err.strip() == "[context] step 2: 25.0% used (2048/8192 input tokens)"


def test_extract_source_urls_from_final_answer_text() -> None:
    output = "Sources: [Example](https://Example.com/report?ref=agent). See also https://docs.example.org/a/b, done."

    assert _extract_source_urls(output) == [
        "https://Example.com/report?ref=agent",
        "https://docs.example.org/a/b",
    ]


def test_normalize_source_url_for_comparison() -> None:
    assert _normalize_source_url("HTTPS://Example.COM:443/report/#section") == "https://example.com/report"
    assert _normalize_source_url("https://example.com/report/?utm=1", drop_query=True) == "https://example.com/report"


def test_validate_cited_sources_allows_visited_urls() -> None:
    _validate_cited_sources_were_visited(
        "Answer citing https://example.com/report.",
        ["https://example.com/report/"],
    )


def test_validate_cited_sources_allows_query_only_differences() -> None:
    _validate_cited_sources_were_visited(
        "Answer citing https://example.com/report.",
        ["https://example.com/report?utm_source=search"],
    )


def test_validate_cited_sources_retries_for_unvisited_urls() -> None:
    with pytest.raises(ModelRetry, match="never navigated to"):
        _validate_cited_sources_were_visited(
            "Answer citing https://example.com/report.",
            ["https://other.example/report"],
        )


async def test_fetch_max_model_len_uses_pydantic_ai_openai_client() -> None:
    model = SimpleNamespace(
        client=SimpleNamespace(
            models=SimpleNamespace(
                list=_AsyncModelList(
                    [
                        _ModelCard("other-model", 4096),
                        _ModelCard("local-model", 8192),
                    ]
                )
            )
        )
    )

    assert await _fetch_max_model_len(model, "local-model") == 8192


async def test_run_agent_iter_drives_nodes_with_hook_safe_next() -> None:
    run = _FakeAgentRun()
    agent = _FakeAgent(run)
    deps = object()
    limits = UsageLimits(request_limit=3)

    result = await _run_agent_iter(agent, "prompt", deps, limits)  # type: ignore[arg-type]

    assert result.output == "done"
    assert agent.prompt == "prompt"
    assert agent.deps is deps
    assert agent.usage_limits is limits
    assert run.next_calls == [run.start_node]
    assert not run.used_bare_iteration


class _AsyncModelList:
    def __init__(self, data: list[object]) -> None:
        self._data = data

    async def __call__(self) -> SimpleNamespace:
        return SimpleNamespace(data=self._data)


class _ModelCard:
    def __init__(self, model_id: str, max_model_len: int) -> None:
        self.id = model_id
        self._max_model_len = max_model_len

    def model_dump(self) -> dict[str, object]:
        return {"id": self.id, "max_model_len": self._max_model_len}


class _FakeAgent:
    def __init__(self, run: "_FakeAgentRun") -> None:
        self._run = run
        self.prompt: str | None = None
        self.deps: object | None = None
        self.usage_limits: UsageLimits | None = None

    def iter(self, prompt: str, *, deps: object, usage_limits: UsageLimits) -> "_FakeAgentRunContext":
        self.prompt = prompt
        self.deps = deps
        self.usage_limits = usage_limits
        return _FakeAgentRunContext(self._run)


class _FakeAgentRunContext:
    def __init__(self, run: "_FakeAgentRun") -> None:
        self._run = run

    async def __aenter__(self) -> "_FakeAgentRun":
        return self._run

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class _FakeAgentRun:
    def __init__(self) -> None:
        self.start_node = object()
        self.next_node = self.start_node
        self.next_calls: list[object] = []
        self.result: AgentRunResult[str] | None = None
        self.used_bare_iteration = False

    def __aiter__(self) -> "_FakeAgentRun":
        self.used_bare_iteration = True
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration

    async def next(self, node: object) -> End[str]:
        self.next_calls.append(node)
        self.result = AgentRunResult("done")
        return End(data="done")

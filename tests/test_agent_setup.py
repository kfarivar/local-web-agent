from types import SimpleNamespace

from pydantic_ai.usage import RunUsage

from efficient_web_agent.agent import (
    _fetch_max_model_len,
    _print_context_usage,
    _usage_to_dict,
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

    assert {"observe_page", "search_page", "extract_neighborhood", "click_element", "screenshot_for_query_help", "websearch"} <= tool_names


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

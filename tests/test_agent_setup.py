from pydantic_ai.usage import RunUsage

from efficient_web_agent.agent import _usage_to_dict, create_agent, create_model
from efficient_web_agent.settings import AgentSettings


def test_create_model_uses_openai_compatible_settings() -> None:
    settings = AgentSettings(model_name="local-model", base_url="http://0.0.0.0:8000/v1", api_key="api-key-not-set")

    model = create_model(settings)

    assert model.model_name == "local-model"


def test_create_agent_registers_context_safe_tools() -> None:
    settings = AgentSettings(model_name="local-model")

    agent = create_agent(settings)
    tool_names = set(agent._function_toolset.tools)  # noqa: SLF001 - verifies public behavior missing a public accessor.

    assert {"observe_page", "search_page", "extract_neighborhood", "click_element", "screenshot_for_query_help"} <= tool_names


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

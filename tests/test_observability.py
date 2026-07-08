from pydantic_ai import Agent

from efficient_web_agent import observability


def test_configure_llm_observability_enables_content_capture(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(observability, "_llm_observability_configured", False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr("langfuse.get_client", lambda: object())
    monkeypatch.setattr(Agent, "instrument_all", lambda instrument: calls.append(instrument))

    observability.configure_llm_observability()

    assert len(calls) == 1
    assert calls[0].include_content is True
    assert calls[0].include_binary_content is False


def test_configure_llm_observability_noops_without_langfuse_keys(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(observability, "_llm_observability_configured", False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(Agent, "instrument_all", lambda instrument: calls.append(instrument))

    observability.configure_llm_observability()

    assert calls == []

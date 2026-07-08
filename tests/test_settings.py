from pathlib import Path

import pytest

from efficient_web_agent.settings import AgentSettings


def test_settings_use_code_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EWA_MODEL_NAME", raising=False)
    monkeypatch.delenv("EWA_MAX_STEPS", raising=False)

    settings = AgentSettings()

    assert settings.model_name == "Qwen/Qwen3-4B-AWQ"
    assert settings.max_steps == 20


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        """
model_name: yaml-model
base_url: http://127.0.0.1:8000/v1
max_steps: 7
vision_enabled: true
max_region_chars: 900
match_neighborhood_chars: 180
""",
        encoding="utf-8",
    )

    settings = AgentSettings.from_yaml(path)

    assert settings.model_name == "yaml-model"
    assert settings.base_url == "http://127.0.0.1:8000/v1"
    assert settings.max_steps == 7
    assert settings.vision_enabled is True
    assert settings.max_region_chars == 900
    assert settings.match_neighborhood_chars == 180


def test_settings_priority_yaml_env_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        """
model_name: yaml-model
max_steps: 3
headless: false
vision_enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("EWA_MODEL_NAME", "env-model")
    monkeypatch.setenv("EWA_MAX_STEPS", "5")
    monkeypatch.setenv("EWA_HEADLESS", "true")

    settings = AgentSettings.from_sources(
        yaml_path=path,
        cli_overrides={"model_name": "cli-model", "vision_enabled": True},
    )

    assert settings.model_name == "cli-model"
    assert settings.max_steps == 5
    assert settings.headless is True
    assert settings.vision_enabled is True


def test_yaml_file_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        AgentSettings.from_yaml(path)

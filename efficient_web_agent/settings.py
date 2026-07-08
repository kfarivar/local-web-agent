"""Runtime settings for the efficient web agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


ENV_PREFIX = "EWA_"
ENV_FIELDS = {
    "model_name": "EWA_MODEL_NAME",
    "base_url": "EWA_BASE_URL",
    "api_key": "EWA_API_KEY",
    "max_steps": "EWA_MAX_STEPS",
    "vision_enabled": "EWA_VISION_ENABLED",
    "snippet_char_budget": "EWA_SNIPPET_CHAR_BUDGET",
    "max_region_chars": "EWA_MAX_REGION_CHARS",
    "match_neighborhood_chars": "EWA_MATCH_NEIGHBORHOOD_CHARS",
    "tool_result_char_budget": "EWA_TOOL_RESULT_CHAR_BUDGET",
    "history_message_limit": "EWA_HISTORY_MESSAGE_LIMIT",
    "headless": "EWA_HEADLESS",
    "aria_depth": "EWA_ARIA_DEPTH",
    "browser_timeout_ms": "EWA_BROWSER_TIMEOUT_MS",
}


class AgentSettings(BaseModel):
    """Runtime configuration for the browser agent.

    Attributes:
        model_name: OpenAI-compatible model name exposed by the local server.
        base_url: OpenAI-compatible API base URL.
        api_key: API key sent to the OpenAI-compatible provider.
        max_steps: Maximum model requests for one agent run.
        vision_enabled: Whether screenshot payloads may be sent to the model.
        snippet_char_budget: Character budget for a full extraction result.
        max_region_chars: Maximum characters for a single returned DOM region.
        match_neighborhood_chars: Context retained around search matches, or
            derived from `max_region_chars` when omitted.
        tool_result_char_budget: Character budget for browser action results.
        history_message_limit: Maximum message count retained by history trimming.
        headless: Whether Camoufox should run without a visible browser window.
        aria_depth: Depth passed to Playwright's AI ARIA snapshot.
        browser_timeout_ms: Default Playwright timeout in milliseconds.
    """

    model_name: str = "Qwen/Qwen3-4B-AWQ"
    base_url: str = "http://0.0.0.0:8000/v1"
    api_key: str = "api-key-not-set"
    max_steps: int = Field(default=20, ge=1)
    vision_enabled: bool = False
    snippet_char_budget: int = Field(default=2500, ge=200)
    max_region_chars: int = Field(default=1200, ge=100)
    match_neighborhood_chars: int | None = Field(default=None, ge=0)
    tool_result_char_budget: int = Field(default=4000, ge=500)
    history_message_limit: int = Field(default=24, ge=4)
    headless: bool = False
    aria_depth: int = Field(default=4, ge=1)
    browser_timeout_ms: int = Field(default=15000, ge=1000)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentSettings":
        """Load settings from a YAML file and built-in defaults only.

        Args:
            path: Path to a YAML mapping whose keys match AgentSettings fields.

        Returns:
            Validated AgentSettings with YAML values overriding code defaults.
        """

        return cls.model_validate(_load_yaml_mapping(path))

    @classmethod
    def from_sources(
        cls,
        *,
        yaml_path: str | Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> "AgentSettings":
        """Load settings with priority: YAML < environment variables < CLI.

        Args:
            yaml_path: Optional YAML file path used as the lowest-priority
                external configuration source.
            cli_overrides: Optional key/value overrides from parsed CLI args.
                Entries with `None` values are ignored.

        Returns:
            Validated AgentSettings after applying all configured sources.
        """

        values: dict[str, Any] = {}
        if yaml_path is not None:
            values.update(_load_yaml_mapping(yaml_path))
        values.update(_environment_overrides())
        if cli_overrides:
            values.update({key: value for key, value in cli_overrides.items() if value is not None})
        return cls.model_validate(values)

    @classmethod
    def from_env(cls) -> "AgentSettings":
        """Load settings from built-in defaults plus environment variables.

        Args:
            None.

        Returns:
            Validated AgentSettings with `EWA_*` variables applied.
        """

        return cls.from_sources()


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read a YAML settings file as a dictionary.

    Args:
        path: YAML file path to open with UTF-8 encoding.

    Returns:
        Top-level YAML mapping as a plain dict. Empty YAML files return an
        empty dict.

    Raises:
        ValueError: If the YAML document is not a top-level mapping.
    """

    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Settings YAML must contain a mapping at the top level: {path}")
    return dict(loaded)


def _environment_overrides() -> dict[str, str]:
    """Collect configured environment-variable overrides.

    Args:
        None.

    Returns:
        Mapping from AgentSettings field names to raw environment string values
        for variables present in `ENV_FIELDS`.
    """

    return {field: os.environ[env_name] for field, env_name in ENV_FIELDS.items() if env_name in os.environ}

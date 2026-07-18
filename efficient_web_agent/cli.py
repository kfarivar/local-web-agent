"""Command-line entrypoint for efficient-web-agent."""

from __future__ import annotations

import argparse
import asyncio
import json

from .agent import run_agent
from .settings import AgentSettings


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the console script.

    Args:
        None.

    Returns:
        Configured ArgumentParser for the `efficient-web-agent` command.
    """

    parser = argparse.ArgumentParser(prog="efficient-web-agent")
    parser.add_argument("goal", help="Browser task to complete.")
    parser.add_argument("--settings", "--config", dest="settings_path", help="YAML settings file to load before env/CLI overrides.")
    parser.add_argument("--model", dest="model_name", help="OpenAI-compatible model name exposed by vLLM.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", help="API key value for the OpenAI-compatible endpoint.")
    parser.add_argument("--max-steps", type=int, help="Maximum model requests/tool steps.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None, help="Run Camoufox headless.")
    parser.add_argument("--vision", action=argparse.BooleanOptionalAction, default=None, help="Allow screenshot payloads for vision-capable local models.")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    """Run the CLI asynchronously.

    Args:
        argv: Optional argument list for tests or programmatic invocation. When
            omitted, argparse reads from `sys.argv`.

    Returns:
        Process-style exit code, where `0` means the agent completed normally.
    """

    args = build_parser().parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "model_name": args.model_name,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "max_steps": args.max_steps,
            "headless": args.headless,
            "vision_enabled": args.vision,
        }.items()
        if value is not None
    }
    settings = AgentSettings.from_sources(yaml_path=args.settings_path, cli_overrides=overrides)
    result = await run_agent(args.goal, settings)

    print('answer:')
    print(result.answer)

    print('\nmeta data:')
    print(json.dumps(result.model_dump(mode="json", exclude={"answer"}), indent=2))

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the async CLI from synchronous entry points.

    Args:
        argv: Optional argument list forwarded to `async_main`.

    Returns:
        Process-style exit code from `async_main`.
    """

    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

"""Small Langfuse helpers with safe defaults for sensitive/large payloads."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_llm_observability_configured = False


def observed(name: str, as_type: str = "span") -> Callable[[F], F]:
    """Decorate a function with Langfuse when configured, otherwise no-op.

    Args:
        name: Langfuse observation name to attach to the wrapped function.
        as_type: Langfuse observation type such as `span`, `tool`, or `agent`.

    Returns:
        Decorator that either applies `langfuse.observe` with content capture
        disabled or returns the original function unchanged.
    """

    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return _identity_decorator()

    try:
        from langfuse import observe

        return observe(name=name, as_type=as_type, capture_input=False, capture_output=False)
    except Exception:
        return _identity_decorator()


def _identity_decorator() -> Callable[[F], F]:
    """Create a decorator that returns functions unchanged.

    Args:
        None.

    Returns:
        No-op decorator used when Langfuse is not configured or import fails.
    """

    def decorator(func: F) -> F:
        """Wrap a function without changing its behavior.

        Args:
            func: Function being decorated.

        Returns:
            Wrapper that forwards all positional and keyword arguments to `func`.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Forward a call to the wrapped function.

            Args:
                *args: Positional arguments for the wrapped function.
                **kwargs: Keyword arguments for the wrapped function.

            Returns:
                Whatever the wrapped function returns.
            """

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def configure_llm_observability() -> None:
    """Enable Pydantic AI model-call spans for Langfuse when configured.

    Args:
        None.

    Returns:
        None. The function silently leaves instrumentation disabled when
        required Langfuse credentials or imports are unavailable.
    """

    global _llm_observability_configured
    if _llm_observability_configured:
        return
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return

    try:
        from langfuse import get_client
        from pydantic_ai import Agent
        from pydantic_ai.models.instrumented import InstrumentationSettings

        get_client()
        Agent.instrument_all(
            InstrumentationSettings(
                include_content=True,
                include_binary_content=False,
            )
        )
        _llm_observability_configured = True
    except Exception:
        return


def flush_observability() -> None:
    """Flush queued Langfuse spans if a client is configured.

    Args:
        None.

    Returns:
        None. Errors are swallowed so observability cannot break agent runs.
    """

    try:
        from langfuse import get_client

        client = get_client()
        flush = getattr(client, "flush", None)
        if callable(flush):
            flush()
    except Exception:
        return

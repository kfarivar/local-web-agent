"""Typed data exchanged between tools, browser code, and the caller."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BrowserState(BaseModel):
    """Bounded browser state returned to the agent.

    Attributes:
        url: Current browser page URL.
        title: Current page title when it can be read.
        aria_snapshot: Bounded accessibility snapshot for the page body.
    """

    url: str
    title: str = ""
    aria_snapshot: str = ""


class ElementRef(BaseModel):
    """Internal reference to an indexed DOM element.

    Attributes:
        ref_id: Stable per-page element identifier such as `e12`.
        tag: HTML tag name for the indexed element.
        role: Inferred or explicit accessibility role.
        name: Label-like metadata from attributes such as `aria-label`.
        text: Full cleaned element text kept locally for search.
        css_path: CSS selector path used to resolve the element in Playwright.
    """

    ref_id: str
    tag: str
    role: str = ""
    name: str = ""
    text: str = ""
    css_path: str


class DomSnippet(BaseModel):
    """Model-facing excerpt from an indexed DOM element.

    Attributes:
        ref_id: Element reference that can be passed to browser action tools.
        role: Inferred or explicit accessibility role.
        name: Label-like metadata for the element when available.
        text: Bounded cleaned snippet text safe to place in model context.
        css_path: CSS selector path for diagnostics and click resolution.
        score: Search relevance score assigned by the page indexer.
    """

    ref_id: str
    role: str = ""
    name: str = ""
    text: str
    css_path: str
    score: float = 0.0


class ExtractionResult(BaseModel):
    """Result returned by DOM search and extraction operations.

    Attributes:
        query: Query, ref_id, or operation label that produced the result.
        matched: Whether at least one snippet was returned.
        snippets: Bounded snippets selected from the local page index.
        truncated: Whether text or result count was reduced to fit budgets.
        char_count: Approximate serialized character count of returned snippets.
    """

    query: str
    matched: bool
    snippets: list[DomSnippet] = Field(default_factory=list)
    truncated: bool = False
    char_count: int = 0


class ActionResult(BaseModel):
    """Outcome from a browser action.

    Attributes:
        ok: Whether the action completed successfully.
        message: Short human-readable action status.
        state: Optional bounded browser state after the action.
    """

    ok: bool
    message: str
    state: BrowserState | None = None


class AgentResult(BaseModel):
    """Final result returned by the public agent runner.

    Attributes:
        answer: Final answer text produced by the agent.
        steps: Count of tool/model interaction steps recorded in dependencies.
        visited_urls: URLs observed during the run.
        usage: Provider or Pydantic AI usage metadata serialized as a dict.
    """

    answer: str
    steps: int = 0
    visited_urls: list[str] = Field(default_factory=list)
    usage: dict[str, object] = Field(default_factory=dict)

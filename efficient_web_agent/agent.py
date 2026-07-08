"""Pydantic AI agent wiring and browser tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Literal

from pydantic_ai import Agent, BinaryContent, RunContext, ToolReturn
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from .browser import BrowserController
from .context import cap_text, history_processor
from .models import ActionResult, AgentResult, BrowserState, ExtractionResult
from .observability import configure_llm_observability, flush_observability, observed
from .settings import AgentSettings

INSTRUCTIONS = """You are an efficient web agent for a local LLM with a small context window.

Core rules:
- Never ask for or expect full webpage dumps.
- use navigate to access webpages.
- Use observe_page, list_interactive, search_page, and extract_neighborhood to inspect bounded page regions.
- Use search_page with focused terms before extracting neighborhoods.
- If repeated search_page calls do not find useful snippets, use screenshot_for_query_help only when vision is enabled.
- Prefer click_element with a DOM ref_id over text guesses. Element clicks use DOM locator/bounding-box data.
- Keep a compact memory of tried queries, visited pages, and useful ref_ids.
- Return the final answer directly when the task is complete.
"""


@dataclass
class AgentDeps:
    """Runtime dependencies shared by Pydantic AI tools.

    Attributes:
        settings: Agent settings for budgets, model, and browser behavior.
        browser: Active BrowserController used by tools.
        visited_urls: Ordered URL log populated during the run.
        failed_searches: Count of consecutive search_page calls with no match.
        steps: Count of tool calls recorded by the agent tools.
    """

    settings: AgentSettings
    browser: BrowserController
    visited_urls: list[str] = field(default_factory=list)
    failed_searches: int = 0
    steps: int = 0


def create_model(settings: AgentSettings) -> OpenAIChatModel:
    """Create the OpenAI-compatible chat model for the local LLM server.

    Args:
        settings: AgentSettings containing model name, base URL, and API key.

    Returns:
        OpenAIChatModel configured with an OpenAIProvider for the local endpoint.
    """

    provider = OpenAIProvider(base_url=settings.base_url, api_key=settings.api_key)
    return OpenAIChatModel(settings.model_name, provider=provider)


def create_agent(settings: AgentSettings) -> Agent[AgentDeps, str]:
    """Create the Pydantic AI web agent and register browser tools.

    Args:
        settings: AgentSettings controlling model configuration, history
            trimming, and tool behavior.

    Returns:
        Agent configured with instructions, dependency type, history processor,
        and the registered web-navigation/extraction tools.
    """

    configure_llm_observability()
    agent = Agent(
        create_model(settings),
        deps_type=AgentDeps,
        instructions=INSTRUCTIONS,
        history_processors=[history_processor(settings.history_message_limit)],
        tool_timeout=30,
    )

    @agent.tool(include_return_schema=False)
    async def observe_page(ctx: RunContext[AgentDeps]) -> BrowserState:
        """Return the current page state without exposing full HTML.

        Args:
            None.

        Returns:
            BrowserState with the current URL, page title, and a bounded
            AI-optimized ARIA snapshot. The result never includes raw full-page
            HTML or full-page text.
        """

        ctx.deps.steps += 1
        await ctx.deps.browser.refresh_index()
        state = await ctx.deps.browser.state()
        _record_url(ctx.deps, state.url)
        return state

    @agent.tool(include_return_schema=False)
    async def navigate(ctx: RunContext[AgentDeps], url: str) -> ActionResult:
        """Navigate the browser to a URL and refresh the page index.

        Args:
            url: Absolute or domain-like URL to open. If no scheme is provided,
                `https://` is added automatically.

        Returns:
            ActionResult indicating success or failure, plus the new bounded
            BrowserState when navigation succeeds.
        """

        ctx.deps.steps += 1
        result = await ctx.deps.browser.navigate(url)
        if result.state is not None:
            _record_url(ctx.deps, result.state.url)
        return _cap_action_result(result, ctx.deps.settings.tool_result_char_budget)

    @agent.tool(include_return_schema=False)
    async def go_back(ctx: RunContext[AgentDeps]) -> ActionResult:
        """Go back one entry in browser history.

        Args:
            None.

        Returns:
            ActionResult indicating whether back navigation succeeded, plus a
            bounded BrowserState for the resulting page when available.
        """

        ctx.deps.steps += 1
        result = await ctx.deps.browser.go_back()
        if result.state is not None:
            _record_url(ctx.deps, result.state.url)
        return _cap_action_result(result, ctx.deps.settings.tool_result_char_budget)

    @agent.tool(include_return_schema=False)
    async def scroll(ctx: RunContext[AgentDeps], direction: Literal["up", "down"], amount: int = 700) -> ActionResult:
        """Scroll the visible page.

        Args:
            direction: Scroll direction, either `up` or `down`.
            amount: Number of pixels to scroll. Use a moderate value when
                searching nearby content and a larger value to move faster.

        Returns:
            ActionResult describing the scroll and the bounded BrowserState
            after scrolling.
        """

        ctx.deps.steps += 1
        return _cap_action_result(await ctx.deps.browser.scroll(direction, amount), ctx.deps.settings.tool_result_char_budget)

    @agent.tool(include_return_schema=False)
    async def list_interactive(ctx: RunContext[AgentDeps], limit: int = 20) -> ExtractionResult:
        """List currently indexed interactive elements.

        Args:
            limit: Maximum number of interactive elements to return. Values are
                clamped to a safe range.

        Returns:
            ExtractionResult containing bounded snippets for links, buttons,
            inputs, and similar controls. Use returned `ref_id` values with
            click_element or type_text.
        """

        ctx.deps.steps += 1
        index = await ctx.deps.browser.refresh_index()
        return index.list_interactive(limit=max(1, min(limit, 50)))

    @agent.tool(include_return_schema=False)
    async def search_page(ctx: RunContext[AgentDeps], query: str, limit: int = 5) -> ExtractionResult:
        """Search the BeautifulSoup page index for relevant page regions.

        Args:
            query: Focused search text, keywords, or phrase to find in the
                cleaned local page index.
            limit: Maximum number of matching snippets to return. Values are
                clamped to a safe range.

        Returns:
            ExtractionResult with ranked, bounded DomSnippet objects. Snippet
            text is compressed around matched terms when possible and never
            contains raw HTML.
        """

        ctx.deps.steps += 1
        index = await ctx.deps.browser.refresh_index()
        result = index.search(query, limit=max(1, min(limit, 20)))
        ctx.deps.failed_searches = 0 if result.matched else ctx.deps.failed_searches + 1
        return result

    @agent.tool(include_return_schema=False)
    async def extract_neighborhood(ctx: RunContext[AgentDeps], ref_id: str, before: int = 1, after: int = 2) -> ExtractionResult:
        """Extract bounded snippets around an indexed DOM ref_id.

        Args:
            ref_id: Element reference returned by search_page or
                list_interactive, such as `e12`.
            before: Number of indexed elements before `ref_id` to include.
            after: Number of indexed elements after `ref_id` to include.

        Returns:
            ExtractionResult containing the requested element and nearby indexed
            elements, bounded to the context budget.
        """

        ctx.deps.steps += 1
        index = ctx.deps.browser.index or await ctx.deps.browser.refresh_index()
        return index.extract_neighborhood(ref_id, before=max(0, min(before, 5)), after=max(0, min(after, 5)))

    @agent.tool(include_return_schema=False)
    async def click_element(ctx: RunContext[AgentDeps], ref_id: str) -> ActionResult:
        """Click an indexed element by DOM ref_id.

        Args:
            ref_id: Element reference returned by search_page or
                list_interactive, such as `e12`. Prefer this over guessing
                visible text or CSS selectors.

        Returns:
            ActionResult describing the click result and the bounded
            BrowserState after the click. The click is performed through the
            DOM locator and element bounding box.
        """

        ctx.deps.steps += 1
        result = await ctx.deps.browser.click_element(ref_id)
        if result.state is not None:
            _record_url(ctx.deps, result.state.url)
        return _cap_action_result(result, ctx.deps.settings.tool_result_char_budget)

    @agent.tool(include_return_schema=False)
    async def type_text(ctx: RunContext[AgentDeps], ref_id: str, text: str, submit: bool = False) -> ActionResult:
        """Type text into an indexed input-like element.

        Args:
            ref_id: Element reference for an input, textarea, select-like, or
                editable control returned by search_page or list_interactive.
            text: Text to enter into the control.
            submit: Whether to press Enter after filling the control.

        Returns:
            ActionResult describing the typing result and the bounded
            BrowserState after the action.
        """

        ctx.deps.steps += 1
        result = await ctx.deps.browser.type_text(ref_id, text, submit)
        if result.state is not None:
            _record_url(ctx.deps, result.state.url)
        return _cap_action_result(result, ctx.deps.settings.tool_result_char_budget)

    @agent.tool(include_return_schema=False)
    async def screenshot_for_query_help(ctx: RunContext[AgentDeps]) -> ToolReturn | str:
        """Capture a viewport screenshot for visual query planning.

        Args:
            None.

        Returns:
            If vision is enabled, a ToolReturn containing a PNG viewport
            screenshot and a short text result. If vision is disabled, a string
            explaining that DOM, ARIA, and BeautifulSoup search tools should be
            used instead.
        """

        ctx.deps.steps += 1
        if not ctx.deps.settings.vision_enabled:
            return "Vision is disabled for this run. Use observe_page, ARIA, list_interactive, or more specific DOM search terms."
        png = await ctx.deps.browser.screenshot()
        return ToolReturn(
            return_value=f"Viewport screenshot captured ({len(png)} bytes). Use it only to choose better DOM/search terms.",
            content=[BinaryContent(data=png, media_type="image/png")],
            metadata={"bytes": len(png), "media_type": "image/png"},
        )

    return agent


@observed("agent.run", as_type="agent")
async def run_agent(goal: str, settings: AgentSettings | None = None) -> AgentResult:
    """Run the browser agent for a single user goal.

    Args:
        goal: Natural-language task to complete with the browser.
        settings: Optional settings. When omitted, defaults plus environment
            variables are loaded through `AgentSettings.from_env`.

    Returns:
        AgentResult containing final answer, step count, visited URLs, and usage
        metadata from Pydantic AI when available.
    """

    settings = settings or AgentSettings.from_env()
    agent = create_agent(settings)
    async with BrowserController(settings) as browser:
        deps = AgentDeps(settings=settings, browser=browser)
        result = await agent.run(
            f"Complete this browser task efficiently with bounded context only:\n\n{goal}",
            deps=deps,
            usage_limits=UsageLimits(request_limit=settings.max_steps),
        )
        flush_observability()
        return AgentResult(
            answer=str(result.output),
            steps=deps.steps,
            visited_urls=deps.visited_urls,
            usage=_usage_to_dict(result.usage()),
        )


def _record_url(deps: AgentDeps, url: str) -> None:
    """Append a URL to dependency state if it differs from the last URL.

    Args:
        deps: AgentDeps object whose `visited_urls` list should be updated.
        url: URL to record.

    Returns:
        None.
    """

    if url and (not deps.visited_urls or deps.visited_urls[-1] != url):
        deps.visited_urls.append(url)


def _cap_action_result(result: ActionResult, limit: int) -> ActionResult:
    """Cap text fields inside an ActionResult.

    Args:
        result: ActionResult returned by BrowserController.
        limit: Character limit applied to the message and ARIA snapshot.

    Returns:
        Copy of `result` with long message and state ARIA text truncated.
    """

    state = result.state
    if state is not None:
        state = state.model_copy(update={"aria_snapshot": cap_text(state.aria_snapshot, limit)[0]})
    return result.model_copy(update={"message": cap_text(result.message, limit)[0], "state": state})


def _usage_to_dict(usage: object | None) -> dict[str, object]:
    """Convert Pydantic AI usage objects to a serializable dictionary.

    Args:
        usage: Usage object returned by Pydantic AI, or None.

    Returns:
        Empty dict for None, `model_dump(mode="json")` output when available,
        dataclass `asdict` output for dataclasses, or `vars` for plain objects.
    """

    if usage is None:
        return {}
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if is_dataclass(usage):
        return asdict(usage)
    return dict(vars(usage))

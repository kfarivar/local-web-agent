"""Pydantic AI agent wiring and browser tools."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, is_dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from .browser import BrowserController
from .context import cap_text, history_processor
from .models import ActionResult, AgentResult, BrowserState, ExtractionResult
from .observability import configure_llm_observability, flush_observability, observed
from .settings import AgentSettings
from .websearch import create_websearch_tool

INSTRUCTIONS = """You are a web agent with a small context window.

Core rules:
- Never use your memory as a source of information. All the info you use should be extracted from the web. reference the source for every fact or figure you use. Only use reliable sources.
- Use search_page with focused terms before extracting neighborhoods.
- Prefer click_element with a DOM ref_id over text guesses. Element clicks use DOM locator/bounding-box data.


The general process to follow:
1. use the websearch tool to find promising urls based on the task goal.
2. use the navigate tool to access each of the urls you have found in step 1.
3. use the observe_page, list_interactive, search_page, and extract_neighborhood tools to inspect bounded page regions and extract the necessary info.

"""


@dataclass
class AgentDeps:
    """Runtime dependencies shared by Pydantic AI tools.

    Attributes:
        settings: Agent settings for budgets, model, and browser behavior.
        browser: Active BrowserController used by tools.
        visited_urls: Ordered URL log populated during the run.
        failed_searches: Count of consecutive search_page calls with no match.
        web_searches: Count of websearch tool calls during the run.
        steps: Count of tool calls recorded by the agent tools.
        max_model_len: Effective model context window reported by the server,
            when available.
        model_requests: Count of completed model requests during the run.
    """

    settings: AgentSettings
    browser: BrowserController
    visited_urls: list[str] = field(default_factory=list)
    failed_searches: int = 0
    web_searches: int = 0
    steps: int = 0
    max_model_len: int | None = None
    model_requests: int = 0


def create_model(settings: AgentSettings) -> OpenAIChatModel:
    """Create the OpenAI-compatible chat model for the local LLM server.

    Args:
        settings: AgentSettings containing model name, base URL, and API key.

    Returns:
        OpenAIChatModel configured with an OpenAIProvider for the local endpoint.
    """

    provider = OpenAIProvider(base_url=settings.base_url, api_key=settings.api_key)
    return OpenAIChatModel(settings.model_name, provider=provider)


def create_agent(settings: AgentSettings, model: OpenAIChatModel | None = None) -> Agent[AgentDeps, str]:
    """Create the Pydantic AI web agent and register browser tools.

    Args:
        settings: AgentSettings controlling model configuration, history
            trimming, and tool behavior.
        model: Optional prebuilt model. When omitted, a model is created from
            settings.

    Returns:
        Agent configured with instructions, dependency type, history processor,
        and the registered web-navigation/extraction tools.
    """

    configure_llm_observability()
    hooks = Hooks()

    @hooks.on.after_model_request
    async def print_context_usage(
        ctx: RunContext[AgentDeps],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        _ = request_context
        ctx.deps.model_requests += 1
        _print_context_usage(
            ctx.deps.model_requests,
            response.usage.input_tokens,
            ctx.deps.max_model_len,
        )
        return response

    agent = Agent(
        model or create_model(settings),
        deps_type=AgentDeps,
        instructions=INSTRUCTIONS,
        tools=[create_websearch_tool(result_char_budget=settings.tool_result_char_budget)],
        capabilities=[ProcessHistory(history_processor(settings.history_message_limit)), hooks],
        tool_timeout=30,
        model_settings={"parallel_tool_calls": False},
        # setting output_type=FinalAnswer will cause the model to stop thinking since it can only respond in tool format. using NativeOutput makes it worse and all the generations will be in the FinalAnswer format which ends the run after 1 step.
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
    model = create_model(settings)
    agent = create_agent(settings, model=model)
    max_model_len = await _fetch_max_model_len(model, settings.model_name)
    async with BrowserController(settings) as browser:
        deps = AgentDeps(settings=settings, browser=browser, max_model_len=max_model_len)
        result = await agent.run(
            f"Complete this browser task efficiently with bounded context only:\n\n{goal}",
            deps=deps,
            usage_limits=UsageLimits(request_limit=settings.max_steps),
        )
        flush_observability()
        usage = result.usage
        if callable(usage):
            usage = usage()
        return AgentResult(
            answer=result.output,
            steps=deps.steps,
            visited_urls=deps.visited_urls,
            usage=_usage_to_dict(usage),
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


async def _fetch_max_model_len(model: OpenAIChatModel, model_name: str) -> int | None:
    try:
        model_list = await model.client.models.list()
    except Exception:
        return None

    models = list(model_list.data)
    matching_models = [item for item in models if item.id == model_name]

    for item in [*matching_models, *models]:
        max_model_len = getattr(item, "max_model_len", None)
        if not isinstance(max_model_len, int):
            model_dump = getattr(item, "model_dump", None)
            if callable(model_dump):
                dumped = model_dump()
                if isinstance(dumped, dict):
                    max_model_len = dumped.get("max_model_len")
        if isinstance(max_model_len, int) and max_model_len > 0:
            return max_model_len
    return None


def _print_context_usage(step: int, input_tokens: int, max_model_len: int | None) -> None:
    if max_model_len is None:
        message = f"[context] step {step}: {input_tokens} input tokens (max context unknown)"
    else:
        percentage = input_tokens / max_model_len * 100
        message = (
            f"[context] step {step}: "
            f"{percentage:.1f}% used ({input_tokens}/{max_model_len} input tokens)"
        )
    print(message, file=sys.stderr, flush=True)


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

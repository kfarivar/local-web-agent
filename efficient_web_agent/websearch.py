"""Extensible web search tool backends for the agent."""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from .context import cap_text

class WebSearchUserLocation(BaseModel):
    """Location hint for web search providers that support localization."""

    city: str | None = Field(default=None, description="City to use when localizing search results.")
    country: str | None = Field(default=None, description="Country to use when localizing search results.")
    region: str | None = Field(default=None, description="Region or state to use when localizing search results.")
    timezone: str | None = Field(default=None, description="Timezone to use when localizing search results.")


class WebSearchConfig(BaseModel):
    """Default web search options, mirroring Pydantic AI WebSearch attributes."""

    user_location: WebSearchUserLocation | None = Field(
        default=None,
        description="Default user location hint for backends that support localized search.",
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="Default domains to exclude from search results.",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Default domains to include in search results.",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        description="Default maximum number of raw search results requested from the backend.",
    )
    max_uses: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of websearch tool calls allowed during one agent run.",
    )


class WebSearchRequest(WebSearchConfig):
    """Search request passed to a backend."""

    query: str = Field(description="Search query to send to the backend.")


class WebSearchResult(BaseModel):
    """Single web search result."""

    title: str = Field(description="Search result title.")
    url: str = Field(description="Search result URL.")
    body: str = Field(description="Search result snippet or summary.")


class WebSearchResponse(BaseModel):
    """Bounded web search response returned to the agent."""

    query: str = Field(description="Query used for the search.")
    backend: str = Field(description="Backend that produced the results.")
    matched: bool = Field(description="Whether any results were returned.")
    results: list[WebSearchResult] = Field(default_factory=list, description="Search results.")
    truncated: bool = Field(default=False, description="Whether result text or result count was truncated.")
    char_count: int = Field(default=0, description="Approximate character count of returned result text.")


class WebSearchBackend(Protocol):
    """Backend protocol for pluggable web search providers."""

    name: str

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """Run a web search request and return bounded results."""


class DuckDuckGoWebSearchBackend:
    """DuckDuckGo implementation backed by Pydantic AI's common search tool."""

    name = "duckduckgo"

    def __init__(self, *, max_results: int | None = 8) -> None:
        """Initialize the backend.

        Args:
            max_results: Maximum raw results requested from DuckDuckGo.
        """

        self._max_results = max_results
        self._tool = duckduckgo_search_tool(max_results=max_results)

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """Run the DuckDuckGo search tool.

        Args:
            request: Search request containing the query and optional filters.

        Returns:
            WebSearchResponse with results normalized to this module's schema.
        """

        tool = self._tool
        if request.max_results < self._max_results:
            tool = duckduckgo_search_tool(max_results=request.max_results)
        raw_results = await tool.function(request.query)
        results = [
            WebSearchResult(title=result["title"], url=result["href"], body=result["body"])
            for result in raw_results
        ]
        return WebSearchResponse(
            query=request.query,
            backend=self.name,
            matched=bool(results),
            results=results,
            char_count=sum(len(result.title) + len(result.url) + len(result.body) for result in results),
        )


def create_websearch_tool(
    *,
    backend: WebSearchBackend | None = None,
    result_char_budget: int = 4000,
    user_location: WebSearchUserLocation | None = None,
    blocked_domains: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    max_results: int | None = 8,
    max_uses: int | None = None,
) -> Tool[Any]:
    """Create the model-facing websearch tool.

    Args:
        backend: Search backend to use. Defaults to DuckDuckGo.
        result_char_budget: Maximum characters returned to the model.
        user_location: Default user-location hint for supporting backends.
        blocked_domains: Default domains excluded from results.
        allowed_domains: Default domains allowed in results.
        max_results: Default maximum number of raw search results requested
            from the backend. This will be enforced over the value the agent sets, 
            unless the agent sets it to a lower value.
        max_uses: Maximum websearch calls during one agent run.

    Returns:
        Pydantic AI Tool that can be registered on an agent.
    """

    default_backend = backend or DuckDuckGoWebSearchBackend(max_results=max_results)
    defaults = WebSearchConfig(
        user_location=user_location,
        blocked_domains=blocked_domains,
        allowed_domains=allowed_domains,
        max_results=max_results,
        max_uses=max_uses,
    )

    async def websearch(
        ctx: RunContext[Any],
        query: str,
        user_location: WebSearchUserLocation | None = None,
        blocked_domains: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        max_results: int | None = None,
    ) -> WebSearchResponse:
        """Search the public web and return bounded result snippets.

        Args:
            query: Search query to send to the configured backend.
            user_location: Location hint for localized search. Local
                DuckDuckGo currently ignores this hint.
            blocked_domains: Domains to exclude from returned results.
            allowed_domains: Domains to include in returned results.
            max_results: Maximum number of raw search results requested from
                the backend.

        Returns:
            WebSearchResponse with normalized title, URL, and snippet results.
        """

        if hasattr(ctx.deps, "steps"):
            ctx.deps.steps += 1

        current_uses = getattr(ctx.deps, "web_searches", 0)
        if defaults.max_uses is not None and current_uses >= defaults.max_uses:
            return WebSearchResponse(
                query=query,
                backend=default_backend.name,
                matched=False,
                results=[],
                char_count=0,
            )
        if hasattr(ctx.deps, "web_searches"):
            ctx.deps.web_searches = current_uses + 1

        request = WebSearchRequest(
            query=query,
            user_location=user_location or defaults.user_location,
            blocked_domains=_merge_domains(defaults.blocked_domains, blocked_domains),
            allowed_domains=defaults.allowed_domains,
            max_results=max_results if max_results is not None else defaults.max_results,
            max_uses=defaults.max_uses,
        )
        response = await default_backend.search(request)
        filtered_results = _filter_domains(
            response.results,
            allowed_domains=request.allowed_domains,
            blocked_domains=request.blocked_domains,
        )
        response = response.model_copy(update={"matched": bool(filtered_results), "results": filtered_results})
        return _cap_response(response, result_char_budget)

    return Tool[Any](websearch, name="websearch", include_return_schema=False)


def _cap_response(response: WebSearchResponse, limit: int) -> WebSearchResponse:
    """Cap a web search response while preserving every full result URL.

    URLs are kept intact first. The remaining character budget is split as
    evenly as possible across all results, with each result spending its share
    on the title before the body.

    TODO: currently the even splitting of budget can lead to results with short body to not use their budget fully. will think about a fix later. 
    """

    if not response.results:
        return response.model_copy(update={"matched": False, "results": [], "char_count": 0})

    url_chars = sum(len(result.url) for result in response.results)
    text_budget = max(0, limit - url_chars)
    per_result_budgets = _split_budget(text_budget, len(response.results))
    capped_results: list[WebSearchResult] = []
    truncated = response.truncated

    for result, result_budget in zip(response.results, per_result_budgets, strict=True):
        title, title_truncated = cap_text(result.title, result_budget)
        body_budget = max(0, result_budget - len(title))
        body, body_truncated = cap_text(result.body, body_budget)
        truncated = truncated or title_truncated or body_truncated
        capped_results.append(result.model_copy(update={"title": title, "body": body}))

    char_count = sum(len(result.title) + len(result.url) + len(result.body) for result in capped_results)
    return response.model_copy(
        update={
            "matched": bool(capped_results),
            "results": capped_results,
            "truncated": truncated,
            "char_count": char_count,
        }
    )


def _split_budget(total: int, count: int) -> list[int]:
    """Split a total character budget as evenly as possible."""

    if count <= 0:
        return []
    base, remainder = divmod(max(0, total), count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _filter_domains(
    results: list[WebSearchResult],
    *,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> list[WebSearchResult]:
    """Apply allowed and blocked domain filters to normalized results."""

    allowed = [_normalize_domain(domain) for domain in allowed_domains or []]
    blocked = [_normalize_domain(domain) for domain in blocked_domains or []]
    filtered: list[WebSearchResult] = []
    for result in results:
        domain = _result_domain(result.url)
        if allowed and not any(_domain_matches(domain, allowed_domain) for allowed_domain in allowed):
            continue
        if blocked and any(_domain_matches(domain, blocked_domain) for blocked_domain in blocked):
            continue
        filtered.append(result)
    return filtered


def _merge_domains(default_domains: list[str] | None, request_domains: list[str] | None) -> list[str] | None:
    """Merge default and request domains while preserving order."""

    merged: list[str] = []
    seen: set[str] = set()
    for domain in [*(default_domains or []), *(request_domains or [])]:
        normalized = _normalize_domain(domain)
        if normalized and normalized not in seen:
            merged.append(domain)
            seen.add(normalized)
    return merged or None


def _result_domain(url: str) -> str:
    """Extract and normalize a result URL hostname."""

    parsed = urlparse(url if "://" in url else f"https://{url}")
    return _normalize_domain(parsed.hostname or "")


def _normalize_domain(domain: str) -> str:
    """Normalize domains for suffix matching."""

    return domain.lower().strip().removeprefix("www.")


def _domain_matches(domain: str, filter_domain: str) -> bool:
    """Return whether a domain equals or is a subdomain of a filter."""

    return domain == filter_domain or domain.endswith(f".{filter_domain}")

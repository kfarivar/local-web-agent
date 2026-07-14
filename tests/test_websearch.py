from __future__ import annotations

from types import SimpleNamespace

from efficient_web_agent.websearch import (
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
    create_websearch_tool,
)


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[WebSearchRequest] = []

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        self.requests.append(request)
        return WebSearchResponse(
            query=request.query,
            backend=self.name,
            matched=True,
            results=[
                WebSearchResult(title="Allowed", url="https://docs.example.com/a", body="a" * 100),
                WebSearchResult(title="Blocked", url="https://blocked.example.com/b", body="b" * 100),
            ],
        )


async def test_websearch_tool_passes_pydantic_ai_style_arguments_to_backend() -> None:
    backend = FakeBackend()
    tool = create_websearch_tool(
        backend=backend,
        result_char_budget=1000,
        blocked_domains=["default-blocked.example.com"],
        max_results=3,
    )
    ctx = SimpleNamespace(deps=SimpleNamespace(steps=0, web_searches=0))

    result = await tool.function(
        ctx,
        query="pydantic ai",
        allowed_domains=["example.com"],
        blocked_domains=["blocked.example.com", "default-blocked.example.com"],
        max_results=5,
    )

    assert ctx.deps.steps == 1
    assert ctx.deps.web_searches == 1
    assert result.backend == "fake"
    assert [search_result.url for search_result in result.results] == ["https://docs.example.com/a"]
    assert backend.requests[0].allowed_domains == ["example.com"]
    assert backend.requests[0].blocked_domains == ["default-blocked.example.com", "blocked.example.com"]
    assert backend.requests[0].max_results == 5


async def test_websearch_tool_uses_constructor_max_results_by_default() -> None:
    backend = FakeBackend()
    tool = create_websearch_tool(backend=backend, max_results=4)
    ctx = SimpleNamespace(deps=SimpleNamespace(steps=0, web_searches=0))

    await tool.function(ctx, query="default max")

    assert backend.requests[0].max_results == 4


async def test_websearch_tool_caps_results_to_budget() -> None:
    backend = FakeBackend()
    tool = create_websearch_tool(backend=backend, result_char_budget=70)
    ctx = SimpleNamespace(deps=SimpleNamespace(steps=0, web_searches=0))

    result = await tool.function(ctx, query="budget")

    assert result.truncated
    assert result.char_count <= 70
    assert [search_result.url for search_result in result.results] == [
        "https://docs.example.com/a",
        "https://blocked.example.com/b",
    ]
    assert [len(search_result.title) + len(search_result.body) for search_result in result.results] == [8, 7]


async def test_websearch_tool_keeps_full_urls_even_when_urls_exceed_budget() -> None:
    backend = FakeBackend()
    tool = create_websearch_tool(backend=backend, result_char_budget=10)
    ctx = SimpleNamespace(deps=SimpleNamespace(steps=0, web_searches=0))

    result = await tool.function(ctx, query="tiny budget")

    assert result.truncated
    assert [search_result.url for search_result in result.results] == [
        "https://docs.example.com/a",
        "https://blocked.example.com/b",
    ]
    assert all(search_result.title == "" and search_result.body == "" for search_result in result.results)


async def test_websearch_tool_respects_max_uses_without_calling_backend() -> None:
    backend = FakeBackend()
    tool = create_websearch_tool(backend=backend, max_uses=1)
    ctx = SimpleNamespace(deps=SimpleNamespace(steps=0, web_searches=1))

    result = await tool.function(ctx, query="over limit")

    assert ctx.deps.steps == 1
    assert ctx.deps.web_searches == 1
    assert not result.matched
    assert backend.requests == []

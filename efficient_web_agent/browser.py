"""Async Camoufox/Playwright browser controller."""

from __future__ import annotations

import asyncio

from .context import cap_text
from .models import ActionResult, BrowserState
from .observability import observed
from .page_index import PageIndexer
from .settings import AgentSettings


class BrowserController:
    """Owns the Camoufox browser page and current BeautifulSoup index.

    Args:
        settings: AgentSettings controlling browser launch, timeouts, ARIA
            depth, and context budgets.
    """

    def __init__(self, settings: AgentSettings) -> None:
        """Initialize controller state without launching the browser.

        Args:
            settings: Runtime settings used by browser and indexing methods.
        """

        self.settings = settings
        self._browser_cm = None
        self.browser = None
        self.page = None
        self.index: PageIndexer | None = None

    async def __aenter__(self) -> "BrowserController":
        """Start the browser when entering an async context manager.

        Args:
            None.

        Returns:
            The started BrowserController instance.
        """

        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close the browser when leaving an async context manager.

        Args:
            exc_type: Exception type from the context body, if any.
            exc: Exception instance from the context body, if any.
            tb: Traceback from the context body, if any.

        Returns:
            None.
        """

        await self.close()

    @observed("browser.start", as_type="span")
    async def start(self) -> None:
        """Launch Camoufox and create an initial blank page.

        Args:
            None.

        Returns:
            None. The controller stores the browser, page, and context manager.
        """

        from camoufox.async_api import AsyncCamoufox

        self._browser_cm = AsyncCamoufox(headless=self.settings.headless, geoip=False)
        self.browser = await self._browser_cm.__aenter__()
        self.page = await self.browser.new_page()
        self.page.set_default_timeout(self.settings.browser_timeout_ms)
        await self.page.goto("about:blank")

    @observed("browser.close", as_type="span")
    async def close(self) -> None:
        """Close the active Camoufox browser context.

        Args:
            None.

        Returns:
            None. Browser, page, and index references are reset.
        """

        if self._browser_cm is not None:
            await self._browser_cm.__aexit__(None, None, None)
        self._browser_cm = None
        self.browser = None
        self.page = None
        self.index = None

    def _require_page(self):
        """Return the active Playwright page or fail if the browser is closed.

        Args:
            None.

        Returns:
            The active Playwright page object.

        Raises:
            RuntimeError: If `start` has not been called or the controller was closed.
        """

        if self.page is None:
            raise RuntimeError("BrowserController has not been started.")
        return self.page

    @observed("browser.state", as_type="span")
    async def state(self) -> BrowserState:
        """Read bounded state from the current page.

        Args:
            None.

        Returns:
            BrowserState with URL, title, and a bounded ARIA snapshot. If ARIA
            capture fails, the snapshot contains a short error marker.
        """

        page = self._require_page()
        title = ""
        aria = ""
        try:
            title = await page.title()
        except Exception:
            title = ""
        try:
            aria = await page.locator("body").aria_snapshot(depth=self.settings.aria_depth, mode="ai", timeout=3000)
            aria = cap_text(aria, self.settings.tool_result_char_budget)[0]
        except Exception as exc:
            aria = f"[ARIA snapshot unavailable: {type(exc).__name__}]"
        return BrowserState(url=page.url, title=title, aria_snapshot=aria)

    @observed("browser.refresh_index", as_type="retriever")
    async def refresh_index(self) -> PageIndexer:
        """Rebuild the local BeautifulSoup index from the current page HTML.

        Args:
            None.

        Returns:
            PageIndexer built from `page.content()` using current context-budget
            settings.
        """

        page = self._require_page()
        html = await page.content()
        self.index = PageIndexer(
            html,
            snippet_char_budget=self.settings.snippet_char_budget,
            max_region_chars=self.settings.max_region_chars,
            match_neighborhood_chars=self.settings.match_neighborhood_chars,
        )
        return self.index

    @observed("browser.navigate", as_type="tool")
    async def navigate(self, url: str) -> ActionResult:
        """Navigate to a URL and refresh the page index.

        Args:
            url: URL or domain-like string. `https://` is prepended when no
                supported scheme is present.

        Returns:
            ActionResult containing a success message and bounded page state.
        """

        page = self._require_page()
        if not url.startswith(("http://", "https://", "about:")):
            url = "https://" + url
        await page.goto(url, wait_until="domcontentloaded", timeout=self.settings.browser_timeout_ms)
        await asyncio.sleep(0.5)
        await self.refresh_index()
        return ActionResult(ok=True, message=f"Navigated to {url}", state=await self.state())

    @observed("browser.go_back", as_type="tool")
    async def go_back(self) -> ActionResult:
        """Navigate one step back in browser history.

        Args:
            None.

        Returns:
            ActionResult containing a status message and bounded page state.
        """

        page = self._require_page()
        await page.go_back(wait_until="domcontentloaded", timeout=self.settings.browser_timeout_ms)
        await asyncio.sleep(0.3)
        await self.refresh_index()
        return ActionResult(ok=True, message="Navigated back", state=await self.state())

    @observed("browser.scroll", as_type="tool")
    async def scroll(self, direction: str, amount: int = 700) -> ActionResult:
        """Scroll the visible page by a pixel amount.

        Args:
            direction: `down` for positive vertical scroll or `up` for negative
                vertical scroll.
            amount: Pixel distance passed to `window.scrollBy`.

        Returns:
            ActionResult describing the scroll or a validation failure.
        """

        page = self._require_page()
        if direction not in {"up", "down"}:
            return ActionResult(ok=False, message="direction must be 'up' or 'down'")
        delta = amount if direction == "down" else -amount
        await page.evaluate("(dy) => window.scrollBy(0, dy)", delta)
        await asyncio.sleep(0.2)
        return ActionResult(ok=True, message=f"Scrolled {direction} {amount}px", state=await self.state())

    @observed("browser.click_element", as_type="tool")
    async def click_element(self, ref_id: str) -> ActionResult:
        """Click an indexed element by ref_id.

        Args:
            ref_id: Element reference from the current PageIndexer.

        Returns:
            ActionResult describing success or failure. Successful clicks use
            the indexed CSS path and click the element center when a bounding
            box is available.
        """

        page = self._require_page()
        index = self.index or await self.refresh_index()
        element = index.get(ref_id)
        if element is None:
            return ActionResult(ok=False, message=f"No indexed element with ref_id={ref_id!r}")

        locator = page.locator(element.css_path).first
        box = await locator.bounding_box(timeout=3000)
        if box is not None:
            await locator.click(position={"x": box["width"] / 2, "y": box["height"] / 2}, timeout=5000)
        else:
            await locator.click(timeout=5000)
        await asyncio.sleep(0.5)
        await self.refresh_index()
        return ActionResult(ok=True, message=f"Clicked {ref_id} ({element.name or element.text[:80]})", state=await self.state())

    @observed("browser.type_text", as_type="tool")
    async def type_text(self, ref_id: str, text: str, submit: bool = False) -> ActionResult:
        """Fill an indexed input-like element and optionally press Enter.

        Args:
            ref_id: Element reference from the current PageIndexer.
            text: Text to fill into the resolved locator.
            submit: Whether to press Enter after filling the locator.

        Returns:
            ActionResult describing success or failure and bounded page state.
        """

        page = self._require_page()
        index = self.index or await self.refresh_index()
        element = index.get(ref_id)
        if element is None:
            return ActionResult(ok=False, message=f"No indexed element with ref_id={ref_id!r}")

        locator = page.locator(element.css_path).first
        await locator.fill(text, timeout=5000)
        if submit:
            await locator.press("Enter", timeout=5000)
            await asyncio.sleep(0.5)
            await self.refresh_index()
        return ActionResult(ok=True, message=f"Typed text into {ref_id}", state=await self.state())

    @observed("browser.screenshot", as_type="tool")
    async def screenshot(self) -> bytes:
        """Capture a PNG screenshot of the current viewport.

        Args:
            None.

        Returns:
            PNG image bytes from Playwright `page.screenshot`.
        """

        page = self._require_page()
        return await page.screenshot(type="png", full_page=False, animations="disabled")

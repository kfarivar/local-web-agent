from efficient_web_agent.browser import BrowserController
from efficient_web_agent.settings import AgentSettings


class FakeLocator:
    def __init__(self) -> None:
        self.clicked = False

    @property
    def first(self):
        return self

    async def bounding_box(self, timeout=None):
        return {"x": 10, "y": 20, "width": 100, "height": 30}

    async def click(self, **kwargs):
        self.clicked = True
        self.kwargs = kwargs


class FakePage:
    url = "https://example.test"

    def __init__(self) -> None:
        self.fake_locator = FakeLocator()
        self.selectors = []

    async def content(self):
        return '<html><body><button id="go">Go</button></body></html>'

    def locator(self, selector):
        self.last_selector = selector
        self.selectors.append(selector)
        return self.fake_locator

    async def title(self):
        return "Example"


async def test_click_element_uses_css_path_and_bounding_box_position() -> None:
    controller = BrowserController(AgentSettings())
    page = FakePage()
    controller.page = page
    await controller.refresh_index()

    result = await controller.click_element("e1")

    assert result.ok
    assert page.selectors[0] == "button#go"
    assert page.fake_locator.clicked
    assert page.fake_locator.kwargs["position"] == {"x": 50.0, "y": 15.0}


async def test_refresh_index_uses_region_settings() -> None:
    controller = BrowserController(AgentSettings(max_region_chars=100, match_neighborhood_chars=25))
    controller.page = FakePage()

    index = await controller.refresh_index()

    assert index.max_region_chars == 100
    assert index.match_neighborhood_chars == 25

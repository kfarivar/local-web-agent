"""BeautifulSoup-backed page indexing with bounded extraction outputs."""

from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, Tag

from .context import cap_text
from .models import DomSnippet, ElementRef, ExtractionResult

NOISE_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "img",
    "picture",
    "source",
    "video",
    "audio",
    "iframe",
    "template",
}
MEANINGFUL_TAGS = {
    "a",
    "button",
    "input",
    "textarea",
    "select",
    "option",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "article",
    "section",
    "main",
    "nav",
    "form",
    "label",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "th",
    "div",
}


def normalize_text(text: str) -> str:
    """Collapse whitespace in text and strip leading/trailing space.

    Args:
        text: Raw text extracted from HTML or attributes.

    Returns:
        Text with consecutive whitespace collapsed to one space.
    """

    return re.sub(r"\s+", " ", text).strip()


def compress_text_around_matches(
    text: str,
    match_terms: Iterable[str],
    *,
    neighborhood_chars: int = 240,
    max_chars: int = 1200,
    marker: str = "[truncated]",
) -> tuple[str, bool]:
    """Keep text around matched terms and replace omitted spans with a marker.

    Args:
        text: Full cleaned element text to compress.
        match_terms: Terms or phrases that should anchor retained text windows.
        neighborhood_chars: Characters to keep on each side of each match
            before enforcing `max_chars`.
        max_chars: Hard maximum length for returned compressed text.
        marker: Text inserted where non-matching spans are omitted.

    Returns:
        Tuple of compressed text and a flag indicating whether any text was
        omitted or hard-capped.
    """

    terms = [term for term in dict.fromkeys(term.strip() for term in match_terms) if term]
    spans: list[tuple[int, int]] = []
    for term in terms:
        for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end()))

    if not spans:
        return cap_text(text, max_chars, marker=f" {marker}")

    def build_with_radius(radius: int) -> tuple[str, bool]:
        """Build compressed text around match spans using a specific radius.

        Args:
            radius: Characters retained on each side of each matched span.

        Returns:
            Tuple of compressed text and whether text outside retained windows
            was omitted.
        """

        windows: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            window_start = max(0, start - radius)
            window_end = min(len(text), end + radius)
            if windows and window_start <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], window_end))
            else:
                windows.append((window_start, window_end))

        parts: list[str] = []
        truncated_text = False
        previous_end = 0
        for start, end in windows:
            if start > previous_end:
                truncated_text = True
                if parts:
                    parts.append(marker)
                elif start > 0:
                    parts.append(marker)
            parts.append(text[start:end].strip())
            previous_end = end
        if previous_end < len(text):
            truncated_text = True
            parts.append(marker)
        return normalize_text(" ".join(part for part in parts if part)), truncated_text

    radius = neighborhood_chars
    compressed, truncated = build_with_radius(radius)
    while len(compressed) > max_chars and radius > 0:
        radius = max(0, radius // 2)
        compressed, truncated = build_with_radius(radius)
    if len(compressed) > max_chars:
        compressed, capped = cap_text(compressed, max_chars, marker=f" {marker}")
        truncated = truncated or capped
    return compressed, truncated


class PageIndexer:
    """Index a webpage without exposing raw full-page content to the model.

    Args:
        html: Raw page HTML to clean and index.
        snippet_char_budget: Maximum serialized character budget for a full extraction result.
        max_region_chars: Maximum text characters retained for any single indexed region/snippet.
        match_neighborhood_chars: Characters retained on each side of matched terms before region capping.
            When omitted, this is derived from `max_region_chars`.
    """

    def __init__(
        self,
        html: str,
        *,
        snippet_char_budget: int = 2500,
        max_region_chars: int = 1200,
        match_neighborhood_chars: int | None = None,
    ) -> None:
        """Parse HTML, remove noisy content, and build the element index.

        Args:
            html: Raw page HTML from the browser.
            snippet_char_budget: Maximum serialized character budget for an
                extraction result.
            max_region_chars: Maximum characters returned for one snippet.
            match_neighborhood_chars: Characters kept around search matches.
                When None, a value is derived from `max_region_chars`.
        """

        self.snippet_char_budget = snippet_char_budget
        self.max_region_chars = max_region_chars
        self.match_neighborhood_chars = (
            max(40, min(max_region_chars // 3, 240)) if match_neighborhood_chars is None else match_neighborhood_chars
        )
        self.soup = BeautifulSoup(html, "html.parser")
        for tag in self.soup.find_all(NOISE_TAGS):
            tag.decompose()
        for tag in self.soup.find_all(attrs={"hidden": True}):
            tag.decompose()
        for tag in self.soup.find_all(attrs={"aria-hidden": "true"}):
            tag.decompose()
        for tag in self.soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            tag.decompose()
        self.elements = self._build_elements()

    def _build_elements(self) -> list[ElementRef]:
        """Build indexed element references from the cleaned BeautifulSoup tree.

        Args:
            None.

        Returns:
            List of ElementRef objects for meaningful tags with useful text or
            interactive semantics.
        """

        elements: list[ElementRef] = []
        root = self.soup.body or self.soup
        for tag in root.find_all(MEANINGFUL_TAGS):
            text = self._element_text(tag)
            if not text and tag.name not in {"input", "textarea", "select", "button", "a"}:
                continue
            if self._is_duplicate_container(tag, text):
                continue
            ref_id = f"e{len(elements) + 1}"
            elements.append(
                ElementRef(
                    ref_id=ref_id,
                    tag=tag.name or "",
                    role=self._role(tag),
                    name=self._name(tag, text),
                    text=text,
                    css_path=self._css_path(tag),
                )
            )
        return elements

    def _element_text(self, tag: Tag) -> str:
        """Extract searchable text from a BeautifulSoup tag.

        Args:
            tag: BeautifulSoup Tag being indexed.

        Returns:
            Normalized text. Input elements use label-like attributes because
            they often do not contain visible child text.
        """

        if tag.name == "input":
            parts = [
                tag.get("aria-label", ""),
                tag.get("placeholder", ""),
                tag.get("value", ""),
                tag.get("name", ""),
                tag.get("type", ""),
            ]
            return normalize_text(" ".join(str(part) for part in parts if part))
        return normalize_text(tag.get_text(" ", strip=True))

    def _is_duplicate_container(self, tag: Tag, text: str) -> bool:
        """Detect container elements that duplicate direct child text.

        Args:
            tag: Candidate BeautifulSoup container tag.
            text: Normalized full text for the candidate tag.

        Returns:
            True when the tag is a structural container whose direct meaningful
            children already represent the same text.
        """

        if tag.name not in {"div", "section", "article", "main", "nav", "form"}:
            return False
        child_texts = [normalize_text(child.get_text(" ", strip=True)) for child in tag.find_all(MEANINGFUL_TAGS, recursive=False)]
        if len(child_texts) < 2:
            return False
        return normalize_text(" ".join(child_texts)) == text

    def _role(self, tag: Tag) -> str:
        """Infer a compact accessibility role for an indexed tag.

        Args:
            tag: BeautifulSoup Tag being indexed.

        Returns:
            Explicit `role` attribute when present, otherwise a small inferred
            role string for common interactive and structural tags.
        """

        if tag.get("role"):
            return str(tag["role"])
        roles = {
            "a": "link",
            "button": "button",
            "input": "textbox",
            "textarea": "textbox",
            "select": "combobox",
            "form": "form",
            "nav": "navigation",
            "main": "main",
            "table": "table",
            "tr": "row",
            "td": "cell",
            "th": "columnheader",
        }
        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return "heading"
        return roles.get(tag.name or "", "")

    def _name(self, tag: Tag, text: str) -> str:
        """Extract a label-like name for an indexed tag.

        Args:
            tag: BeautifulSoup Tag being indexed.
            text: Normalized visible text for the tag. Currently unused except
                for API symmetry with older implementations.

        Returns:
            Normalized value from `aria-label`, `alt`, `title`, `placeholder`,
            or `name`, otherwise an empty string.
        """

        for attr in ("aria-label", "alt", "title", "placeholder", "name"):
            if tag.get(attr):
                return normalize_text(str(tag[attr]))
        return ""

    def _css_path(self, tag: Tag) -> str:
        """Build a CSS selector path for a BeautifulSoup tag.

        Args:
            tag: BeautifulSoup Tag being indexed.

        Returns:
            CSS selector path using IDs when available, limited class names, and
            `nth-of-type` for same-tag siblings.
        """

        parts: list[str] = []
        node: Tag | None = tag
        while node and isinstance(node, Tag) and node.name not in {"[document]", None}:
            if node.get("id"):
                parts.append(f"{node.name}#{self._css_escape(str(node['id']))}")
                break
            selector = node.name or ""
            if node.get("class"):
                classes = ".".join(self._css_escape(str(cls)) for cls in node.get("class", [])[:2])
                if classes:
                    selector = f"{selector}.{classes}"
            sibling_index = 1
            for sibling in node.find_previous_siblings(node.name):
                sibling_index += 1
            if sibling_index > 1:
                selector = f"{selector}:nth-of-type({sibling_index})"
            parts.append(selector)
            node = node.parent if isinstance(node.parent, Tag) else None
        return " > ".join(reversed(parts))

    def _css_escape(self, value: str) -> str:
        """Escape non-selector-safe characters for simple CSS path fragments.

        Args:
            value: ID or class string to escape.

        Returns:
            CSS-fragment string with non-alphanumeric characters backslash escaped.
        """

        return re.sub(r"([^a-zA-Z0-9_-])", r"\\\1", value)

    def get(self, ref_id: str) -> ElementRef | None:
        """Look up an indexed element by reference ID.

        Args:
            ref_id: Element reference such as `e12`.

        Returns:
            Matching ElementRef, or None when the ref_id is not in the index.
        """

        return next((element for element in self.elements if element.ref_id == ref_id), None)

    def search(self, query: str, *, limit: int = 5) -> ExtractionResult:
        """Search indexed elements and return bounded snippets.

        Args:
            query: Search text used for scoring and match-centered compression.
            limit: Maximum number of matching elements considered for the result.

        Returns:
            ExtractionResult containing ranked snippets compressed around query
            matches when possible.
        """

        terms = [term.lower() for term in re.findall(r"\w+", query)]
        snippets: list[tuple[DomSnippet, bool]] = []
        for element in self.elements:
            haystack = " ".join([element.role, element.name, element.text]).lower()
            score = self._score(haystack, query.lower(), terms)
            if score <= 0:
                continue
            snippets.append(self._snippet(element, score=score, match_terms=[query, *terms]))
        snippets.sort(key=lambda snippet: snippet[0].score, reverse=True)
        return self._bounded_result(query, snippets[:limit])

    def extract_neighborhood(self, ref_id: str, *, before: int = 1, after: int = 2) -> ExtractionResult:
        """Return bounded snippets around a specific indexed element.

        Args:
            ref_id: Element reference to center the neighborhood on.
            before: Number of indexed elements before `ref_id` to include.
            after: Number of indexed elements after `ref_id` to include.

        Returns:
            ExtractionResult containing nearby snippets, or an unmatched result
            when the ref_id is not found.
        """

        index = next((i for i, element in enumerate(self.elements) if element.ref_id == ref_id), None)
        if index is None:
            return ExtractionResult(query=ref_id, matched=False, snippets=[])
        start = max(0, index - before)
        end = min(len(self.elements), index + after + 1)
        snippets = [self._snippet(element, score=1.0) for element in self.elements[start:end]]
        return self._bounded_result(ref_id, snippets)

    def list_interactive(self, *, limit: int = 20) -> ExtractionResult:
        """Return bounded snippets for indexed interactive elements.

        Args:
            limit: Maximum number of interactive snippets to return.

        Returns:
            ExtractionResult containing links, buttons, textboxes, comboboxes,
            and similar elements from the current index.
        """

        interactive_roles = {"link", "button", "textbox", "combobox", "checkbox", "radio"}
        snippets = [
            self._snippet(element, score=1.0)
            for element in self.elements
            if element.role in interactive_roles or element.tag in {"a", "button", "input", "textarea", "select"}
        ]
        return self._bounded_result("interactive-elements", snippets[:limit])

    def _score(self, haystack: str, query: str, terms: Iterable[str]) -> float:
        """Score an indexed element against a search query.

        Args:
            haystack: Lowercase searchable text built from role, name, and text.
            query: Lowercase full query string.
            terms: Individual lowercase query terms.

        Returns:
            Numeric relevance score. Full-query matches receive a larger boost
            than individual term matches.
        """

        score = 0.0
        if query and query in haystack:
            score += 5.0
        for term in terms:
            if term in haystack:
                score += 1.0
        return score

    def _snippet(
        self,
        element: ElementRef,
        *,
        score: float,
        match_terms: Iterable[str] | None = None,
    ) -> tuple[DomSnippet, bool]:
        """Create a model-facing snippet for an indexed element.

        Args:
            element: ElementRef selected from the local index.
            score: Relevance score to attach to the snippet.
            match_terms: Optional terms used to compress text around matches.
                When omitted, text is capped from the start.

        Returns:
            Tuple of DomSnippet and a flag indicating whether snippet text was
            truncated or compressed.
        """

        if match_terms:
            text, truncated = compress_text_around_matches(
                element.text,
                match_terms,
                neighborhood_chars=self.match_neighborhood_chars,
                max_chars=self.max_region_chars,
            )
        else:
            text, truncated = cap_text(element.text, self.max_region_chars)
        return (
            DomSnippet(
                ref_id=element.ref_id,
                role=element.role,
                name=element.name,
                text=text,
                css_path=element.css_path,
                score=score,
            ),
            truncated,
        )

    def _bounded_result(self, query: str, snippets: list[tuple[DomSnippet, bool]]) -> ExtractionResult:
        """Fit snippets into the result-level character budget.

        Args:
            query: Query or operation label to attach to the result.
            snippets: Candidate snippets paired with per-snippet truncation flags.

        Returns:
            ExtractionResult containing snippets that fit the budget, plus
            truncation and character-count metadata.
        """

        kept: list[DomSnippet] = []
        char_count = 0
        truncated = False
        for snippet, snippet_truncated in snippets:
            truncated = truncated or snippet_truncated
            serialized = f"{snippet.ref_id} {snippet.role} {snippet.name} {snippet.text} {snippet.css_path}"
            if kept and char_count + len(serialized) > self.snippet_char_budget:
                truncated = True
                break
            if len(serialized) > self.snippet_char_budget:
                text, was_truncated = cap_text(snippet.text, max(0, self.snippet_char_budget - 200))
                snippet = snippet.model_copy(update={"text": text})
                truncated = truncated or was_truncated
                serialized = f"{snippet.ref_id} {snippet.role} {snippet.name} {snippet.text} {snippet.css_path}"
            kept.append(snippet)
            char_count += len(serialized)
        return ExtractionResult(
            query=query,
            matched=bool(kept),
            snippets=kept,
            truncated=truncated,
            char_count=char_count,
        )

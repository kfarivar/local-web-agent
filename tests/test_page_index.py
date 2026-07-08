from efficient_web_agent.page_index import PageIndexer


HTML = """
<html>
  <head><title>Ignored</title><script>secret full dump token</script></head>
  <body>
    <nav><a href="/home">Home</a></nav>
    <main>
      <h1>Search results</h1>
      <section>
        <article>
          <h2><a href="/alpha">Alpha project</a></h2>
          <p>Alpha has 42 stars and a compact description.</p>
        </article>
        <article hidden>
          <h2>Hidden result</h2>
          <p>This should not be indexed.</p>
        </article>
      </section>
      <form><input name="q" placeholder="Search query"></form>
    </main>
  </body>
</html>
"""


def test_search_returns_bounded_clean_snippets_not_raw_page() -> None:
    index = PageIndexer(HTML, snippet_char_budget=240)

    result = index.search("Alpha stars", limit=10)

    assert result.matched
    joined = " ".join(snippet.text for snippet in result.snippets)
    assert "Alpha" in joined
    assert "secret full dump token" not in joined
    assert "<article>" not in joined
    assert result.char_count <= 240


def test_search_matches_text_beyond_return_snippet_cap() -> None:
    html = f"""
    <html><body>
      <article>{"filler " * 80} late_unique_marker appears after the return cap</article>
    </body></html>
    """
    index = PageIndexer(html, snippet_char_budget=300, max_region_chars=80)

    result = index.search("late_unique_marker", limit=1)

    assert result.matched
    assert "late_unique_marker" in result.snippets[0].text
    assert result.snippets[0].text.startswith("[truncated]")
    assert result.truncated


def test_search_snippet_drops_irrelevant_middle_text_between_distant_matches() -> None:
    html = f"""
    <html><body>
      <article>alpha {"middle " * 80} omega</article>
    </body></html>
    """
    index = PageIndexer(html, snippet_char_budget=500, max_region_chars=120)

    result = index.search("alpha omega", limit=1)

    assert result.matched
    snippet = result.snippets[0].text
    assert "alpha" in snippet
    assert "omega" in snippet
    assert "[truncated]" in snippet
    assert len(snippet) <= 120

def test_my_custom_cases_for_text_compression():
    html = f"""
    <html><body>
      <article>alpha {"middle " * 80} alpha{"middle " * 80} omegaalpha {"middle " * 80} omega</article>
    </body></html>
    """
    max_region_chars = 1200
    index = PageIndexer(html, snippet_char_budget=1000, max_region_chars=max_region_chars)

    result = index.search("alpha omega", limit=1)

    assert result.matched
    snippet = result.snippets[0].text

    print(len(snippet))
    print(snippet)

    assert "alpha" in snippet
    assert "omega" in snippet
    assert "[truncated]" in snippet
    assert len(snippet) <= max_region_chars

    



def test_hidden_content_is_not_indexed() -> None:
    index = PageIndexer(HTML)

    result = index.search("should not be indexed")

    assert not result.matched


def test_interactive_elements_include_stable_refs_and_css_paths() -> None:
    index = PageIndexer(HTML)

    result = index.list_interactive()

    refs = {snippet.ref_id: snippet for snippet in result.snippets}
    assert any(snippet.role == "link" and "Home" in snippet.text for snippet in refs.values())
    assert any(snippet.role == "textbox" and "Search query" in snippet.text for snippet in refs.values())
    assert all(snippet.css_path for snippet in refs.values())


def test_name_uses_label_like_attributes_without_repeating_body_text() -> None:
    html = """
    <html><body>
      <button aria-label="Save changes">Save</button>
      <p>Plain paragraph body text.</p>
    </body></html>
    """
    index = PageIndexer(html)

    button = next(element for element in index.elements if element.tag == "button")
    paragraph = next(element for element in index.elements if element.tag == "p")

    assert button.name == "Save changes"
    assert paragraph.name == ""
    assert paragraph.text == "Plain paragraph body text."


def test_extract_neighborhood_uses_ref_id() -> None:
    index = PageIndexer(HTML)
    alpha = index.search("Alpha project").snippets[0]

    result = index.extract_neighborhood(alpha.ref_id, before=1, after=1)

    assert result.matched
    assert any(snippet.ref_id == alpha.ref_id for snippet in result.snippets)

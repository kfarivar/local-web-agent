# Efficient Web Agent

`efficient_web_agent` is a Pydantic AI + Camoufox browser agent designed for local LLMs with small context windows.

The key invariant is that full webpage HTML or text is never sent directly to the model. Pages are parsed locally with BeautifulSoup and exposed to the model through bounded search, extraction, ARIA, action-result, and optional screenshot tools.

Run a task with:

```bash
efficient-web-agent "Find the current page title for example.com"
```

The vLLM-compatible endpoint and Langfuse service are expected to already be running and configured.

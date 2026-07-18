# Efficient Web Agent

A context-efficient browser agent powered by a small local LLM.

Efficient Web Agent combines **Pydantic AI**, **Camoufox/Playwright**, **BeautifulSoup**, and an **OpenAI-compatible local model server** to complete web tasks without dumping entire pages into the model context. It was built for small-context local models, where every tool result needs to be intentional, bounded, and easy for the model to act on.

> TODO: Add a short terminal screenshot showing a successful run, including the context usage lines and final JSON metadata.

## Why This Exists

One of the main reasons browser agents or MCPs often fail locally is because they fill the context with lots of unnecessary info. Full HTML, long tool dumps, and oversized screenshots quickly crowd out space for useful info and reasoning, especially on compact local models.

This project takes a different approach. Instead of giving the model access to the full page the model can get the page summary using the summary agent. To further process a page the web agent can search the page with beautifulsoup to get the top results or request the info in the neighborhood of an element in the HTML.   

- Parse webpages locally.
- Return only bounded, ranked snippets to the model.
- Click and type through stable per-page element references.
- Preserve enough browser state for action, without exposing raw full-page HTML.
- Keep model history trimmed so long runs remain inside the available context window.

## What It Can Do

- Search the web with a bounded [DDGS](https://github.com/deedy5/ddgs)-backed `websearch` tool.
- Navigate real pages with Camoufox.
- Inspect pages through ARIA snapshots and local DOM indexing.
- Search page content with BeautifulSoup-powered snippet extraction.
- Extract neighborhoods around specific DOM references.
- List interactive controls such as links, buttons, textboxes, and selects.
- Click or type into indexed elements by `ref_id`.
- Optionally send viewport screenshots to vision-capable models.
- Report visited URLs, tool steps, and model usage metadata after each run.
- Observability visualization using OTEL and Langfuse.

## Architecture

> TODO placeholder: create a polished architecture graphic.

The core invariant is simple: **the model never needs full webpage HTML or full-page text**. The browser controller owns the live page, the indexer owns the parsed page representation, and the agent receives only compact tool outputs that are sized for the configured budgets.

## Tech Stack

- **Python 3.14**
- **Pydantic AI** for agent orchestration, tool registration, and usage limits
- **Camoufox / Playwright** for browser automation
- **BeautifulSoup** for local HTML parsing and searchable page indexing
- **vLLM-compatible OpenAI API** for local model serving
- **DuckDuckGo(DDGS) search** through Pydantic AI common tools
- **Langfuse** for optional observability
- **pytest** and **pytest-asyncio** for tests
- **uv** for dependency and environment management

## Repository Layout

```text
.
├── efficient_web_agent/
│   ├── agent.py              # Pydantic AI agent wiring and browser tools
│   ├── browser.py            # Camoufox/Playwright browser controller
│   ├── page_index.py         # BeautifulSoup DOM indexing and bounded search
│   ├── websearch.py          # Pluggable bounded web search backend
│   ├── context.py            # Text caps and history trimming
│   ├── settings.py           # YAML/env/CLI configuration
│   ├── observability.py      # Optional Langfuse integration
│   └── models.py             # Typed tool/result models
├── tests/                    # Unit tests for indexing, tools, settings, and observability
├── efficient_web_agent/settings.example.yaml
├── vllm_config.yaml          # Example local model server config
├── run_vllm.sh               # Local convenience launcher for vLLM
├── pyproject.toml
└── uv.lock
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

The project is packaged with a console script named `efficient-web-agent`.

### 2. Start an OpenAI-compatible local model server

This repository includes an example vLLM config for `Qwen/Qwen3-4B-AWQ`:

```bash
vllm serve --config vllm_config.yaml
```

There is also a convenience script:

```bash
./run_vllm.sh
```

`run_vllm.sh` assumes a local vLLM virtual environment path. If your setup differs, update the activation line or run `vllm serve --config vllm_config.yaml` from your own vLLM environment.

### 3. Run a browser task

```bash
uv run efficient-web-agent \
  "Find the current page title for example.com" \
  --settings efficient_web_agent/settings.example.yaml
```

The CLI prints the final answer first, followed by structured run metadata such as step count, visited URLs, and model usage.

## Example Task

```bash
uv run --env-file .env efficient-web-agent \
  "find the main modes of transportation in tokyo then summarize the main info and statistics for each mode. only use the information from sources online do not use your own memory. reference the source for every fact you write." \
  --settings ./efficient_web_agent/settings.example.yaml
```

> Image placeholder: Add a GIF or screenshot sequence showing the browser opening, navigating between sources, and returning the final cited answer.

## Configuration

Settings can come from three places, with later sources overriding earlier ones:

1. YAML config file
2. `EWA_*` environment variables
3. CLI flags

Common CLI overrides:

```bash
efficient-web-agent "your task" \
  --model Qwen/Qwen3-4B-AWQ \
  --base-url http://0.0.0.0:8000/v1 \
  --api-key api-key-not-set \
  --max-steps 20 \
  --headless \
  --no-vision
```

Supported environment variables include:

```text
EWA_MODEL_NAME
EWA_BASE_URL
EWA_API_KEY
EWA_MAX_STEPS
EWA_VISION_ENABLED
EWA_SNIPPET_CHAR_BUDGET
EWA_MAX_REGION_CHARS
EWA_MATCH_NEIGHBORHOOD_CHARS
EWA_TOOL_RESULT_CHAR_BUDGET
EWA_HISTORY_MESSAGE_LIMIT
EWA_HEADLESS
EWA_ARIA_DEPTH
EWA_BROWSER_TIMEOUT_MS
```

See [efficient_web_agent/settings.example.yaml](efficient_web_agent/settings.example.yaml) for a complete example.

## Context Management Design

The agent is designed around context discipline:

- `observe_page` returns URL, title, and a bounded ARIA snapshot.
- `search_page` searches the local BeautifulSoup index and returns ranked snippets.
- `extract_neighborhood` expands around a known `ref_id` without dumping the page.
- `click_element` and `type_text` operate on those references instead of guessed selectors.
- `history_processor` trims old Pydantic AI messages while preserving valid tool-call structure.
- Tool results are capped with explicit character budgets.

This lets the model work from small, high-signal observations while the Python runtime keeps the full browser and DOM state locally.

## Observability

Langfuse instrumentation is optional and safe by default. If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present, the project records agent, browser, retriever, and tool spans. The custom observation decorators avoid capturing large or sensitive browser payloads for most tool spans.

During a run, the agent also prints context usage to stderr when the model endpoint reports `max_model_len`:

```text
[context] step 2: 25.0% used (2048/8192 input tokens)
```

## Testing

Run the test suite with:

```bash
uv run pytest
```

The tests cover:

- Settings loading and override priority
- Text capping and history trimming
- BeautifulSoup page indexing and snippet compression
- Hidden/noisy content filtering
- Interactive element references and CSS paths
- Browser click behavior through indexed elements
- Agent setup, tool registration, and usage serialization
- Web search result filtering and budget caps


## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

run:
	uv run --env-file .env efficient-web-agent 'navigate to the wikipedia page for tokyo. find the 2025 population of tokyo.' --settings ./efficient_web_agent/settings.example.yaml

test:
	uv run --env-file .env python test.py
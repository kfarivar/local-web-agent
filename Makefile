run:
	uv run --env-file .env efficient-web-agent 'find the main modes of transportation in tokyo then summarize the main info and statistics for each mode.' --settings ./efficient_web_agent/settings.example.yaml

test:
	uv run --env-file .env python scratch_test.py

parks-test:
	uv run --env-file .env efficient-web-agent 'navigate to https://parks.canada.ca/voyage-travel/region/ontario/randonnee-hike. for each item extract the location then navigate to https://www.google.com/maps and search for it. after finding it use the share button and get the share link. return all the links.' --settings ./efficient_web_agent/settings.example.yaml
	
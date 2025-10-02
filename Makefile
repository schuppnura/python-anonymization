format:
	black .

lint:
	ruff .

test:
	pytest -q

typecheck:
	mypy .

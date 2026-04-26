run:
	uv run python -m src

install:
	uv pip install -r pyproject.toml

debug:
	uv run python -m pdb src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	flake8 . --exclude=.venv
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

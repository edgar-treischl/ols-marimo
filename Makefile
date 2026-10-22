.PHONY: install check build serve test clean

install:
	uv sync --locked

check:
	uv run marimo check app.py

build: check
	rm -rf dist
	uv run marimo export html-wasm app.py -o dist --mode run

serve: build
	uv run python -m http.server 8000 --directory dist

test: install check build

clean:
	rm -rf dist
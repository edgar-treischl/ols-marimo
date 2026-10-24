.PHONY: install check build serve dev test clean

install:
	uv sync --locked

dev:
	uv run marimo edit app.py --watch --sandbox --host 127.0.0.1 --port 2718

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
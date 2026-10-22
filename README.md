# ols-marimo

Run the browser-based app:

```bash
make serve
# http://localhost:8000
```

The browser downloads the Python runtime and packages on first load, so an
internet connection is required. Data files in `public/` are copied into the
export; use `mo.notebook_location()` to resolve them as local paths in Python
and HTTP URLs in the browser.

Edit the notebook locally:

```bash
uv run marimo edit app.py -- --log-level debug
# http://localhost:2718
```
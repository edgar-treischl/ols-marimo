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
Read HTTP data into a buffer before passing it to pandas: GitHub Pages may
gzip responses, which the browser already decompresses. Passing the URL
directly to pandas can cause a second decompression and a `BadGzipFile` error.

Edit the notebook locally:

```bash
uv run marimo edit app.py -- --log-level debug
# http://localhost:2718
```
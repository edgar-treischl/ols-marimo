# Copilot Instructions for ols-marimo

## Project Overview

**ols-marimo** is a marimo-based data science application that provides an interactive regression explorer. The app loads built-in datasets (Diabetes and Iris), allows users to select variables and perform linear regression analysis with detailed statistical output, visualizations, and model diagnostics. It's built as a marimo notebook (`app.py`) that exports to a web-based, browser-runnable HTML+WASM application.

## Build and Development Commands

### Setup
```bash
make install          # Install dependencies using uv (locked versions)
```

### Development and Testing
```bash
make check            # Run marimo validation on app.py
make build            # Run checks, then export app to dist/ directory
make serve            # Build and start HTTP server on port 8000
make test             # Run check and build (full verification pipeline)
make clean            # Remove dist/ directory
```

### Single Tasks
- **Check syntax only**: `uv run marimo check app.py`
- **Export without serving**: `uv run marimo export html-wasm app.py -o dist --mode run`
- **Custom Python execution**: `uv run python <script>`

## Technology Stack

- **Framework**: marimo (reactive notebook framework)
- **Package Manager**: uv with locked dependencies (uv.lock)
- **Python Version**: ≥3.11 (specified in pyproject.toml), ≥3.12 (in app.py script header)
- **Key Dependencies**:
  - `marimo`: Reactive notebook environment
  - `pandas`: Data manipulation
  - `altair`: Declarative visualization (version ≥6.3.0)
  - Additional via app.py: numpy, scikit-learn, statsmodels, plotly
- **Deployment**: GitHub Pages (via `.github/workflows/deploy.yml`)

## Key Architecture & Patterns

### Marimo Reactive Cell Structure
The app uses marimo's **reactive dependency graph** model:

1. **Cell 0** - Imports: Sets up marimo, numpy, pandas, plotly, scikit-learn, statsmodels
2. **Cell 1** - Data Loading: Loads diabetes and iris datasets from sklearn
3. **Cell 2** - Dataset Picker: UI dropdown in sidebar to select dataset
4. **Cell 3** - Column Selection: Extracts numeric columns from selected dataset
5. **Cell 4** - Variable Pickers: Two dropdowns (X and Y) with intelligent defaults (e.g., "size_m2" or "price_k" if available)
6. **Cell 5** - Data Preparation: Filters selected columns, removes missing values
7. **Cell 6** - Descriptive Statistics: Displays summary stats, missing value counts, and preview table
8. **Cell 7** - Scatterplot: Plotly scatter visualization of X vs Y
9. **Cell 8** - Regression Analysis: Runs OLS model via statsmodels, computes confidence intervals
10. **Cell 9** - Fitted Line Plot: Visualizes observed data with regression line
11. **Cell 10** - UI Layout: Combines all components into tabs (Descriptive statistics & Linear regression)

### Reactivity Rules
- Cells execute automatically when dependencies change (no manual triggers)
- Variables cannot be redeclared in different cells (scope is global)
- UI element values accessed via `.value` attribute in separate cells from definition
- Underscored function names (e.g., `_()`) mark local-only cells
- Last expression in a cell is automatically displayed
- The dependency graph must be acyclic (DAG)

### State & Conditional Logic
- UI elements (dropdowns, etc.) trigger cascading re-executions
- `mo.callout(..., kind='warn')` displays validation messages when:
  - Same variable selected for X and Y
  - Fewer than 3 complete observations
  - Independent variable has < 2 distinct values
- Missing data filtered silently in data prep; counts shown in stats

## Code Conventions

### Cell Editing Guidelines
1. Only edit code **within** the `@app.cell` decorator body
2. Marimo auto-manages function parameters and return statements
3. Always return variables needed by downstream cells (except cell 10, which terminates display)
4. Import all dependencies in Cell 0 (use `import marimo as mo` consistently)
5. Never use `global` declarations

### Visualization Conventions
- **Plotly**: Return figure object directly; use `mo.ui.plotly(fig)` for display
- **Layout**: Use `mo.hstack()`, `mo.vstack()`, and `mo.tabs()` for organization
- **Styling**: Apply templates (e.g., `template="plotly_white"`), set margins and height for consistency

### UI Patterns
- Group related controls in sidebar with `mo.sidebar([...])` 
- Access UI values only in cells *after* the definition cell
- Use descriptive labels for dropdowns and sliders
- Prefix defaults intelligently (e.g., "Diabetes" dataset checks for "size_m2" column)

### Data Handling
- Use pandas for data loading and manipulation
- Apply `.dropna()` to remove missing values before analysis
- Preserve original data for diagnostics (e.g., missing value counts)
- Use `.select_dtypes(include="number")` to filter numeric columns

### Statistical Analysis
- Use `statsmodels.api.OLS` for regression (already imported as `sm`)
- Always use `sm.add_constant()` with `has_constant="add"` for intercept
- Extract confidence intervals via `.conf_int()`
- Display p-values with `model.f_pvalue` (use `:.4g` format for large/small numbers)
- Provide full `model.summary().as_text()` output for transparency

## Workflow

### For Small Updates (bug fixes, style tweaks)
1. Edit relevant cell in app.py
2. Run `make check` to validate
3. Run `make build` to verify export

### For Feature Additions
1. Plan new cell locations in the dependency chain (don't break DAG)
2. Add required imports in Cell 0 if needed
3. Ensure all variables used in new cells are returned by upstream cells
4. Test with `make test`
5. Manual browser test via `make serve` if possible

### Before Pushing
- Run `make test` to ensure check and build pass
- Verify no circular dependencies (marimo will catch)
- If deploying: workflow will automatically export and push to GitHub Pages on merge to main

## Common Pitfalls

1. **Redeclaring variables**: Cannot reuse same variable name in different cells
2. **Accessing UI values in definition cell**: Move UI value access to next cell
3. **Breaking DAG**: Don't create circular cell dependencies
4. **Forgetting returns**: Downstream cells won't have access to undeclared variables
5. **Hardcoding column names**: Use dynamic selection or conditional defaults

## Deployment

GitHub Actions workflow (`.github/workflows/deploy.yml`) automatically:
- Installs uv and Python 3.12
- Runs `uv sync --locked` 
- Exports app with `marimo export html-wasm app.py -o dist --mode run`
- Deploys dist/ to GitHub Pages on push to main

No manual deployment needed; builds happen automatically on merge.

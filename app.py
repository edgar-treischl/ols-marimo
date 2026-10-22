# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "statsmodels",
#     "plotly",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from src.helpers import (
        extract_numeric_columns,
        fit_regression_model,
        load_penguins_data,
        prepare_analysis_data,
        validate_regression_inputs,
    )
    from src.tabs import (
        build_assumptions_tab,
        build_results_tab,
        build_summary_tab,
    )

    return (
        build_assumptions_tab,
        build_results_tab,
        build_summary_tab,
        extract_numeric_columns,
        fit_regression_model,
        load_penguins_data,
        mo,
        prepare_analysis_data,
        validate_regression_inputs,
    )


@app.cell
def _(extract_numeric_columns, load_penguins_data, mo):
    _data_path = mo.notebook_location() / "public" / "data" / "penguins.csv"
    penguins = load_penguins_data(_data_path)

    numeric_columns = extract_numeric_columns(penguins)
    df = penguins.dropna(subset=numeric_columns).copy()
    return df, numeric_columns


@app.cell
def _(df, mo, numeric_columns):
    import pandas as pd

    # Numeric variables and categorical variables eligible for dummy encoding
    categorical_columns = df.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()

    independent_vars = numeric_columns + categorical_columns

    y_picker = mo.ui.dropdown(
        options=numeric_columns,
        value=numeric_columns[1]
        if len(numeric_columns) > 1
        else numeric_columns[0] if numeric_columns else None,
        label="Outcome:",
        full_width=True,
    )
    return independent_vars, y_picker


@app.cell
def _(independent_vars, mo, y_picker):
    # X can be any variable except the selected Y
    available_x_vars = [v for v in independent_vars if v != y_picker.value]

    x_picker = mo.ui.dropdown(
        options=available_x_vars,
        value=available_x_vars[0] if available_x_vars else None,
        label="X Variable",
        full_width=True,
    )


    mo.sidebar(
    [
        mo.md("Select variables."),
        y_picker,
        x_picker,
    ],
    width="320px",)
    return (x_picker,)


@app.cell
def _(df, prepare_analysis_data, x_picker, y_picker):
    x_name = x_picker.value
    y_name = y_picker.value

    analysis_df = prepare_analysis_data(df, x_name, y_name)
    return analysis_df, x_name, y_name


@app.cell
def _(analysis_df, build_summary_tab, mo, x_name, y_name):
    summary_tab = build_summary_tab(mo, analysis_df, x_name, y_name)
    return (summary_tab,)


@app.cell
def _(
    analysis_df,
    fit_regression_model,
    validate_regression_inputs,
    x_name,
    y_name,
):
    model = None

    is_valid, _ = validate_regression_inputs(analysis_df, x_name, y_name)
    if is_valid:
        model = fit_regression_model(analysis_df, x_name, y_name)
    return (model,)


@app.cell
def _(analysis_df, build_results_tab, mo, model, x_name, y_name):
    results_tab = build_results_tab(mo, analysis_df, model, x_name, y_name)
    return (results_tab,)


@app.cell
def _(analysis_df, build_assumptions_tab, mo, model, x_name, y_name):
    assumptions_with_plot = build_assumptions_tab(
        mo,
        analysis_df,
        model,
        x_name,
        y_name,
    )
    return (assumptions_with_plot,)


@app.cell
def _(assumptions_with_plot, mo, results_tab, summary_tab):
    tabs = mo.ui.tabs({
        "Data & Summary": summary_tab,
        "Assumptions": assumptions_with_plot,
        "Results": results_tab,
    })

    mo.vstack([
        mo.md("# Linear Regression Explorer"),
        mo.md(
            "Analyze penguin data relationships through interactive "
            "regression analysis."
        ),
        tabs,
    ], gap=2)
    return


if __name__ == "__main__":
    app.run()

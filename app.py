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

__generated_with = "0.19.0"
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
        mo,
        extract_numeric_columns,
        fit_regression_model,
        load_penguins_data,
        prepare_analysis_data,
        validate_regression_inputs,
        build_assumptions_tab,
        build_results_tab,
        build_summary_tab,
    )


@app.cell
def _(
    mo,
    load_penguins_data,
    extract_numeric_columns,
):
    _data_path = mo.notebook_location() / "public" / "data" / "penguins.csv"
    penguins = load_penguins_data(_data_path)

    numeric_columns = extract_numeric_columns(penguins)
    df = penguins.dropna(subset=numeric_columns).copy()

    return df, numeric_columns


@app.cell
def _(df, mo, numeric_columns):
    x_picker = mo.ui.dropdown(
        options=numeric_columns,
        value=numeric_columns[0],
        label="Independent variable (X)",
        full_width=True,
    )

    default_y = (
        numeric_columns[1]
        if len(numeric_columns) > 1
        else numeric_columns[0]
    )

    y_picker = mo.ui.dropdown(
        options=numeric_columns,
        value=default_y,
        label="Dependent variable (Y)",
        full_width=True,
    )

    mo.sidebar([
        mo.md("## Linear Regression"),
        mo.md("Select variables and explore relationships."),
        x_picker,
        y_picker,
    ])

    return x_picker, y_picker


@app.cell
def _(df, x_picker, y_picker, prepare_analysis_data):
    x_name = x_picker.value
    y_name = y_picker.value

    analysis_df = prepare_analysis_data(df, x_name, y_name)

    return analysis_df, x_name, y_name


@app.cell
def _(mo, analysis_df, x_name, y_name, build_summary_tab):
    summary_tab = build_summary_tab(mo, analysis_df, x_name, y_name)

    return (summary_tab,)


@app.cell
def _(analysis_df, x_name, y_name, validate_regression_inputs, fit_regression_model):
    model = None

    is_valid, _ = validate_regression_inputs(analysis_df, x_name, y_name)
    if is_valid:
        model = fit_regression_model(analysis_df, x_name, y_name)

    return model


@app.cell
def _(mo, analysis_df, model, x_name, y_name, build_results_tab):
    results_tab = build_results_tab(mo, analysis_df, model, x_name, y_name)

    return (results_tab,)


@app.cell
def _(mo, analysis_df, model, x_name, y_name, build_assumptions_tab):
    assumptions_with_plot = build_assumptions_tab(
        mo,
        analysis_df,
        model,
        x_name,
        y_name,
    )

    return (assumptions_with_plot,)


@app.cell
def _(
    assumptions_with_plot,
    mo,
    results_tab,
    summary_tab,
):
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

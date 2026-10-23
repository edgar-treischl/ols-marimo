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
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Regression in a Nutshell
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    First, run the preparation steps below. "
        "The import and data-loading cell runs even when hidden.
    """)
    return


@app.cell
def import_data():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import statsmodels.api as sm
    from IPython.display import HTML
    from plotly.subplots import make_subplots
    from scipy import stats

    import altair as alt

    import matplotlib.pyplot as plt

    from src.helpers import (
        extract_numeric_columns,
        fit_regression_model,
        load_penguins_data,
        prepare_analysis_data,
        validate_regression_inputs,
        scatter_plot,
    )

    data_path = Path("public/data/penguins.csv")
    penguins = load_penguins_data(data_path)

    numeric_columns = extract_numeric_columns(penguins)
    df = penguins.dropna(subset=numeric_columns).copy()
    return (
        df,
        fit_regression_model,
        go,
        make_subplots,
        mo,
        np,
        pd,
        prepare_analysis_data,
        px,
        scatter_plot,
        sm,
        stats,
        validate_regression_inputs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect the data
    """)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df, prepare_analysis_data, validate_regression_inputs):
    # Configure the regression
    y_name = "body_mass_g"
    x_name = "bill_length_mm"

    analysis_df = prepare_analysis_data(df, x_name, y_name)

    is_valid, message = validate_regression_inputs(
        analysis_df, x_name, y_name
    )

    if not is_valid:
        raise ValueError(message)

    print(f"Predictor: {x_name}")
    print(f"Outcome: {y_name}")
    print(f"Observations: {len(analysis_df)}")

    analysis_df.head()
    return analysis_df, x_name, y_name


@app.cell
def _(analysis_df, x_name, y_name):
    # Descriptive statistics
    analysis_df[[x_name, y_name]].describe().T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 Linear Association
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a linear regression, we estimate the effect of an independent variable X on a dependent variable Y. Often, we use several independent variables to predict Y, but let’s stick to the simple model with two variables. The principals of a linear regression are the same.

    So, what happens when we regress X on Y? We calculate the linear association between X and Y and we try to fit a line in accordance to our data. We can use a scatter plot to check the linear association. Let’s have a look for the chosen variables:
    """)
    return


@app.cell
def _(analysis_df, scatter_plot, x_name, y_name):
    scatter_plot(analysis_df, x=x_name, y=y_name)
    return


@app.cell
def _(analysis_df, x_name, y_name):
    # Pearson correlation
    correlation = analysis_df[[x_name, y_name]].corr().iloc[0, 1]
    print(f"Pearson correlation: {correlation:.4f}")
    return


@app.cell
def _(analysis_df, fit_regression_model, pd, x_name, y_name):
    # Fit the regression model
    model = fit_regression_model(analysis_df, x_name, y_name)
    confidence_intervals = model.conf_int()

    coefficients = pd.DataFrame(
        {
            "Term": model.params.index,
            "Coefficient": model.params.values,
            "Std. Error": model.bse.values,
            "t-statistic": model.tvalues.values,
            "p-value": model.pvalues.values,
            "95% CI lower": confidence_intervals.iloc[:, 0].values,
            "95% CI upper": confidence_intervals.iloc[:, 1].values,
        }
    )

    print(f"R²:             {model.rsquared:.3f}")
    print(f"Adjusted R²:    {model.rsquared_adj:.3f}")
    print(f"F-test p-value: {model.f_pvalue:.4g}")
    print(f"Observations:   {int(model.nobs)}")

    coefficients.round(6)
    return (model,)


@app.cell
def _(model):
    # Full regression report with HTML formatting
    print(model.summary())
    return


@app.cell
def _(analysis_df, px, x_name, y_name):
    # Scatter plot and fitted regression line
    fig = px.scatter(
        analysis_df,
        x=x_name,
        y=y_name,
        title=f"{y_name} vs. {x_name}",
        trendline="ols",
        labels={
            x_name: x_name,
            y_name: y_name,
        },
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
    )

    fig
    return


@app.cell
def _(go, make_subplots, model, np):
    # Residual diagnostics
    fitted_values = np.asarray(model.fittedvalues)
    residuals = np.asarray(model.resid)

    fig_resid = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Residuals vs. Fitted",
            "Residual distribution",
        ),
    )

    fig_resid.add_trace(
        go.Scatter(
            x=fitted_values,
            y=residuals,
            mode="markers",
            name="Residuals",
        ),
        row=1,
        col=1,
    )

    fig_resid.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        row=1,
        col=1,
    )

    fig_resid.add_trace(
        go.Histogram(
            x=residuals,
            name="Residuals",
        ),
        row=1,
        col=2,
    )

    fig_resid.update_layout(
        template="plotly_white",
        showlegend=False,
        title_text="Regression diagnostics",
    )

    fig_resid
    return (residuals,)


@app.cell
def _(model, np, residuals, sm, stats):
    # Normality and heteroscedasticity tests
    if 3 <= len(residuals) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)

        print("Shapiro-Wilk normality test")
        print(f"Statistic: {shapiro_stat:.4f}")
        print(f"P-value: {shapiro_p:.4g}")
    else:
        print(
            "Shapiro-Wilk test skipped: "
            "requires 3–5000 observations."
        )

    exog = np.asarray(model.model.exog)

    bp_lm, bp_pvalue, bp_f, bp_f_pvalue = (
        sm.stats.diagnostic.het_breuschpagan(
            residuals, exog
        )
    )

    print("\nBreusch-Pagan heteroscedasticity test")
    print(f"LM statistic: {bp_lm:.4f}")
    print(f"P-value: {bp_pvalue:.4g}")
    return


if __name__ == "__main__":
    app.run()

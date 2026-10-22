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
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import statsmodels.api as sm
    from scipy import stats

    return go, mo, np, pd, sm, stats


@app.cell
def _(mo, pd):
    _data_path = mo.notebook_location() / "public" / "data" / "penguins.csv"
    penguins = pd.read_csv(str(_data_path))

    numeric_columns = penguins.select_dtypes(include="number").columns.tolist()

    if not numeric_columns:
        raise ValueError("The dataset contains no numeric columns.")

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
def _(df, x_picker, y_picker):
    x_name = x_picker.value
    y_name = y_picker.value

    analysis_df = (
        df[[x_name, y_name]]
        .replace([float("inf"), -float("inf")], float("nan"))
        .dropna()
        .copy()
    )

    return analysis_df, x_name, y_name


@app.cell
def _(analysis_df, mo, x_name, y_name):
    if x_name == y_name:
        summary_tab = mo.callout(
            "Please select different variables for X and Y.",
            kind="warn",
        )
    elif analysis_df.empty:
        summary_tab = mo.callout(
            "No valid observations are available for the selected variables.",
            kind="warn",
        )
    else:
        summary = analysis_df.describe().T
        summary.index.name = "Variable"
        summary = summary.reset_index()

        summary_tab = mo.vstack([
            mo.md("## Data Summary"),
            mo.hstack([
                mo.stat(
                    label="Observations",
                    value=str(len(analysis_df)),
                ),
                mo.stat(
                    label="Variables",
                    value="2",
                ),
            ], justify="start"),
            mo.md(f"**X = `{x_name}`** | **Y = `{y_name}`**"),
            mo.md("### Summary Statistics"),
            mo.ui.table(summary.round(4)),
            mo.md("### Data Preview"),
            mo.ui.table(analysis_df.head(15)),
        ])

    return (summary_tab,)


@app.cell
def _(analysis_df, mo, pd, sm, x_name, y_name):
    model = None

    if x_name == y_name:
        results_tab = mo.callout(
            "Select different variables for X and Y.",
            kind="warn",
        )
    elif len(analysis_df) < 3:
        results_tab = mo.callout(
            "At least three valid observations are required.",
            kind="warn",
        )
    elif analysis_df[x_name].nunique() < 2:
        results_tab = mo.callout(
            "The independent variable must contain at least two distinct values.",
            kind="warn",
        )
    elif analysis_df[y_name].nunique() < 2:
        results_tab = mo.callout(
            "The dependent variable is constant; regression statistics are not meaningful.",
            kind="warn",
        )
    else:
        X = sm.add_constant(
            analysis_df[[x_name]],
            has_constant="add",
        )
        y_values = analysis_df[y_name]

        model = sm.OLS(y_values, X).fit()

        confidence_intervals = model.conf_int()

        coefficients = pd.DataFrame({
            "Term": model.params.index,
            "Coefficient": model.params.values,
            "Std. Error": model.bse.values,
            "t-statistic": model.tvalues.values,
            "p-value": model.pvalues.values,
            "95% CI lower": confidence_intervals.iloc[:, 0].values,
            "95% CI upper": confidence_intervals.iloc[:, 1].values,
        })

        intercept = model.params["const"]
        slope = model.params[x_name]

        results_tab = mo.vstack([
            mo.md("## Regression Results"),
            mo.md(
                f"**{y_name} = {intercept:.4f} "
                f"+ ({slope:.4f} × {x_name})**"
            ),
            mo.hstack([
                mo.stat(
                    label="R²",
                    value=f"{model.rsquared:.4f}",
                ),
                mo.stat(
                    label="Adjusted R²",
                    value=f"{model.rsquared_adj:.4f}",
                ),
                mo.stat(
                    label="F-test p-value",
                    value=f"{model.f_pvalue:.4g}",
                ),
                mo.stat(
                    label="Observations",
                    value=str(int(model.nobs)),
                ),
            ], justify="start", gap=1),
            mo.md("### Coefficients & Confidence Intervals"),
            mo.ui.table(coefficients.round(6)),
            mo.md("### Full OLS Summary"),
            mo.plain(model.summary().as_text()),
        ])

    return model, results_tab


@app.cell
def _(analysis_df, go, model, mo, np, stats, x_name, y_name):
    if model is None:
        assumptions_with_plot = mo.vstack([
            mo.md("### Assumptions Checks"),
            mo.md(
                "Visualizations will appear after a valid model is estimated."
            ),
        ])
    else:
        residuals = np.asarray(model.resid)
        fitted_values = np.asarray(model.fittedvalues)

        # Observed values and fitted regression line.
        scatter_fig = go.Figure()

        scatter_fig.add_trace(go.Scatter(
            x=analysis_df[x_name],
            y=analysis_df[y_name],
            mode="markers",
            name="Observed",
            marker=dict(size=7, opacity=0.65),
        ))

        regression_x = np.linspace(
            analysis_df[x_name].min(),
            analysis_df[x_name].max(),
            100,
        )

        regression_y = (
            model.params["const"]
            + model.params[x_name] * regression_x
        )

        scatter_fig.add_trace(go.Scatter(
            x=regression_x,
            y=regression_y,
            mode="lines",
            name="Fitted line",
            line=dict(color="#E45756", width=3),
        ))

        scatter_fig.update_layout(
            title=f"{y_name} vs {x_name} with Fitted Line",
            xaxis_title=x_name,
            yaxis_title=y_name,
            template="plotly_white",
            height=400,
            margin=dict(l=30, r=20, t=60, b=30),
        )

        # Residuals versus fitted values.
        residual_fig = go.Figure()

        residual_fig.add_trace(go.Scatter(
            x=fitted_values,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(size=6, opacity=0.6),
        ))

        residual_fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
        )

        residual_fig.update_layout(
            title="Residuals vs Fitted Values",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            template="plotly_white",
            height=400,
            margin=dict(l=30, r=20, t=60, b=30),
        )

        # Normal Q-Q plot using correct theoretical normal quantiles.
        theoretical_quantiles, ordered_residuals = stats.probplot(
            residuals,
            dist="norm",
            fit=False,
        )

        theoretical_quantiles = np.asarray(theoretical_quantiles)
        ordered_residuals = np.asarray(ordered_residuals)

        qq_fig = go.Figure()

        qq_fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=ordered_residuals,
            mode="markers",
            name="Residuals",
            marker=dict(size=6, opacity=0.6),
        ))

        # Reference line based on the fitted normal distribution.
        qq_slope, qq_intercept, _ = stats.probplot(
            residuals,
            dist="norm",
            fit=True,
        )[1]

        qq_x = np.array([
            theoretical_quantiles.min(),
            theoretical_quantiles.max(),
        ])

        qq_fig.add_trace(go.Scatter(
            x=qq_x,
            y=qq_intercept + qq_slope * qq_x,
            mode="lines",
            name="Normal reference line",
            line=dict(color="red", dash="dash"),
        ))

        qq_fig.update_layout(
            title="Normal Q-Q Plot of Residuals",
            xaxis_title="Theoretical Normal Quantiles",
            yaxis_title="Ordered Residuals",
            template="plotly_white",
            height=400,
            margin=dict(l=30, r=20, t=60, b=30),
        )

        assumptions_with_plot = mo.vstack([
            mo.md("### Scatter Plot with Fitted Line"),
            mo.ui.plotly(scatter_fig),
            mo.md("### Residuals vs Fitted Values"),
            mo.ui.plotly(residual_fig),
            mo.md("### Normal Q-Q Plot"),
            mo.ui.plotly(qq_fig),
        ])

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

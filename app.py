
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "statsmodels",
#     "plotly",
#     "altair",
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
    import plotly.express as px
    import plotly.graph_objects as go
    import statsmodels.api as sm

    from sklearn.datasets import load_diabetes, load_iris

    return (
        go,
        load_diabetes,
        load_iris,
        mo,
        np,
        pd,
        px,
        sm,
    )


@app.cell
def _(load_diabetes, load_iris, np, pd):
    # Dataset 1: Diabetes progression
    diabetes = load_diabetes(as_frame=True)
    diabetes_df = diabetes.frame.copy()

    # Dataset 2: Iris measurements
    iris = load_iris(as_frame=True)
    iris_df = iris.frame.copy()
    iris_df["species"] = [
        iris.target_names[i] for i in iris.target
    ]

    # Dataset 3: Synthetic housing data
    rng = np.random.default_rng(42)
    n = 300

    size = rng.normal(150, 40, n).clip(40, 300)
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 80, n)

    housing_df = pd.DataFrame({
        "size_m2": size,
        "bedrooms": bedrooms,
        "age_years": age,
        "price_k": (
            45
            + 2.8 * size
            + 12 * bedrooms
            - 0.7 * age
            + rng.normal(0, 35, n)
        ),
    })

    datasets = {
        "Diabetes": diabetes_df,
        "Iris": iris_df,
        "Synthetic housing": housing_df,
    }

    return (datasets,)


@app.cell
def _(datasets, mo):
    dataset_picker = mo.ui.dropdown(
        options=list(datasets.keys()),
        value="Synthetic housing",
        label="Dataset",
        full_width=True,
    )

    mo.sidebar([
        mo.md("## Regression Explorer"),
        mo.md("Choose your dataset and variables."),
        dataset_picker,
    ])

    return (dataset_picker,)


@app.cell
def _(dataset_picker, datasets):
    df = datasets[dataset_picker.value].copy()

    numeric_columns = (
        df.select_dtypes(include="number")
        .columns.tolist()
    )

    return df, numeric_columns


@app.cell
def _(mo, numeric_columns):
    # Set defaults based on the selected dataset.
    x_default = (
        "size_m2"
        if "size_m2" in numeric_columns
        else numeric_columns[0]
    )

    y_default = (
        "price_k"
        if "price_k" in numeric_columns
        else numeric_columns[-1]
    )

    x_picker = mo.ui.dropdown(
        options=numeric_columns,
        value=x_default,
        label="Independent variable (X)",
        full_width=True,
    )

    y_picker = mo.ui.dropdown(
        options=numeric_columns,
        value=y_default,
        label="Dependent variable (Y)",
        full_width=True,
    )

    mo.sidebar([
        mo.md("### Model variables"),
        x_picker,
        y_picker,
    ])

    return x_picker, y_picker


@app.cell
def _(df, x_picker, y_picker):
    x_name = x_picker.value
    y_name = y_picker.value

    # Keep the original data available for missing-value counts.
    missing_counts = df[[x_name, y_name]].isna().sum()

    # Use complete observations for analysis.
    analysis_df = df[[x_name, y_name]].dropna().copy()

    return analysis_df, missing_counts, x_name, y_name


@app.cell
def _(analysis_df, missing_counts, mo, x_name, y_name):
    if x_name == y_name:
        descriptive_view = mo.vstack([
            mo.md("## Descriptive statistics"),
            mo.callout(
                "Please select different variables for X and Y.",
                kind="warn",
            ),
        ])
    else:
        summary = analysis_df.describe().T
        summary.index.name = "Variable"
        summary = summary.reset_index()

        missing_table = missing_counts.rename(
            "Missing observations"
        ).rename_axis("Variable").reset_index()

        descriptive_view = mo.vstack([
            mo.md("## Descriptive statistics"),
            mo.hstack([
                mo.stat(
                    label="Complete observations",
                    value=str(len(analysis_df)),
                ),
                mo.stat(
                    label="Selected variables",
                    value="2",
                ),
            ], justify="start"),
            mo.md(
                f"**Dataset columns:** X = `{x_name}`, "
                f"Y = `{y_name}`"
            ),
            mo.md("### Summary statistics"),
            mo.ui.table(summary.round(4)),
            mo.md("### Missing observations before cleaning"),
            mo.ui.table(missing_table),
            mo.md("### Data preview"),
            mo.ui.table(analysis_df.head(10)),
        ])

    return (descriptive_view,)


@app.cell
def _(analysis_df, mo, px, x_name, y_name):
    if x_name == y_name:
        scatter_view = mo.md(
            "Select two different variables to display a scatterplot."
        )
    else:
        scatter_fig = px.scatter(
            analysis_df,
            x=x_name,
            y=y_name,
            title=f"{y_name} versus {x_name}",
            template="plotly_white",
            opacity=0.75,
        )

        scatter_fig.update_layout(
            height=450,
            margin=dict(l=30, r=20, t=60, b=30),
        )

        scatter_view = mo.ui.plotly(scatter_fig)

    return (scatter_view,)


@app.cell
def _(analysis_df, mo, pd, sm, x_name, y_name):
    model = None

    if x_name == y_name:
        regression_view = mo.vstack([
            mo.md("## Linear regression results"),
            mo.callout(
                "Choose different variables for X and Y.",
                kind="warn",
            ),
        ])

    elif len(analysis_df) < 3:
        regression_view = mo.callout(
            "At least three complete observations are required.",
            kind="warn",
        )

    elif analysis_df[x_name].nunique() < 2:
        regression_view = mo.callout(
            "The independent variable must have at least "
            "two distinct values.",
            kind="warn",
        )

    else:
        X = sm.add_constant(
            analysis_df[[x_name]],
            has_constant="add",
        )
        y = analysis_df[y_name]

        model = sm.OLS(y, X).fit()

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

        regression_view = mo.vstack([
            mo.md("## Linear regression results"),
            mo.md(
                f"**Dependent variable:** `{y_name}`  \n"
                f"**Independent variable:** `{x_name}`"
            ),
            mo.md("### Estimated regression equation"),
            mo.md(
                f"`{y_name} = {intercept:.4f} "
                f"+ ({slope:.4f} × {x_name})`"
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
            mo.md("### Coefficients and confidence intervals"),
            mo.ui.table(coefficients.round(4)),
            mo.md("### Full OLS statistical summary"),
            mo.plain(model.summary().as_text()),
        ])

    return model, regression_view


@app.cell
def _(analysis_df, go, model, mo, x_name, y_name):
    if model is None:
        regression_plot = mo.md(
            "The fitted regression plot will appear here "
            "when a valid model can be estimated."
        )
    else:
        regression_x = analysis_df[x_name].sort_values()

        regression_y = (
            model.params["const"]
            + model.params[x_name] * regression_x
        )

        regression_fig = go.Figure()

        regression_fig.add_trace(go.Scatter(
            x=analysis_df[x_name],
            y=analysis_df[y_name],
            mode="markers",
            name="Observed",
            marker=dict(size=7, opacity=0.65),
        ))

        regression_fig.add_trace(go.Scatter(
            x=regression_x,
            y=regression_y,
            mode="lines",
            name="Fitted OLS line",
            line=dict(color="#E45756", width=3),
        ))

        regression_fig.update_layout(
            title="Observed values and fitted regression line",
            xaxis_title=x_name,
            yaxis_title=y_name,
            template="plotly_white",
            height=450,
            margin=dict(l=30, r=20, t=60, b=30),
        )

        regression_plot = mo.ui.plotly(regression_fig)

    return (regression_plot,)


@app.cell
def _(
    descriptive_view,
    mo,
    regression_plot,
    regression_view,
    scatter_view,
):
    tabs = mo.ui.tabs({
        "Descriptive statistics": mo.vstack([
            descriptive_view,
            scatter_view,
        ], gap=2),
        "Linear regression": mo.vstack([
            regression_view,
            regression_plot,
        ], gap=2),
    })

    mo.vstack([
        mo.md("# Regression Explorer"),
        mo.md(
            "Explore built-in datasets, inspect descriptive "
            "statistics, and estimate a simple linear regression."
        ),
        tabs,
    ], gap=2)

    return


if __name__ == "__main__":
    app.run()

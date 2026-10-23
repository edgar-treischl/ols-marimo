"""Tab content builders for the Linear Regression Explorer."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def build_summary_tab(mo, data, x_name, y_name):
    """Build the Data & Summary tab content."""
    if data.empty:
        return mo.callout(
            "No valid observations are available for the selected variables.",
            kind="warn",
        )

    summary = data.describe().T
    summary.index.name = "Variable"
    summary = summary.reset_index()

    return mo.vstack([
        mo.md("## Data Summary"),
        mo.hstack(
            [
                mo.stat(
                    label="Observations",
                    value=str(len(data)),
                ),
                mo.stat(
                    label="Variables",
                    value=str(len(data.columns)),
                ),
            ],
            justify="start",
        ),
        mo.md(f"**X = `{x_name}`** | **Y = `{y_name}`**"),
        mo.md("### Summary Statistics"),
        mo.ui.table(summary.round(4)),
    ])


def dataframe_to_html_table(mo, df):
    """Render a DataFrame as a styled, static HTML table."""
    formatted = df.copy()

    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda x: f"{x:.6f}" if pd.notna(x) else "—"
            )

    return mo.Html(
        formatted.to_html(
            index=False,
            escape=True,
            border=0,
            classes=["regression-table"],
        ).replace(
            '<table ',
            '<table style="'
            'width:100%; border-collapse:collapse; '
            'font-size:0.9rem;" '
        ).replace(
            '<th>',
            '<th style="text-align:left; padding:8px 12px; '
            'border-bottom:2px solid #ccc;">'
        ).replace(
            '<td>',
            '<td style="padding:8px 12px; '
            'border-bottom:1px solid #ddd;">'
        )
    )


def build_results_tab(mo, analysis_df, model, x_name, y_name):
    """Build the Results tab content with regression statistics.
    
    Args:
        mo: marimo module.
        analysis_df: DataFrame with selected analysis data.
        model: Fitted statsmodels OLS model (or None if validation failed).
        x_name: Name of independent variable.
        y_name: Name of dependent variable.
        
    Returns:
        marimo UI component for results tab.
    """
    if model is None:
        return mo.callout(
            "Select different variables for X and Y.",
            kind="warn",
        )
    
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
    
    # Handle both numeric and categorical X
    is_numeric_x = pd.api.types.is_numeric_dtype(analysis_df[x_name])
    
    if is_numeric_x:
        # For numeric X, the predictor column is "_x_numeric"
        slope = model.params["_x_numeric"]
        equation = (
            f"**{y_name} = {intercept:.4f} "
            f"+ ({slope:.4f} × {x_name})**"
        )
    else:
        equation = (
            f"**{y_name} = {intercept:.4f} "
            f"+ [dummy-coded {x_name}]**"
        )
    
    return mo.vstack([
        mo.md("## Regression Results"),
        mo.md(equation),
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
        #dataframe_to_html_table(mo, coefficients),
        mo.md("### Full OLS Summary"),
        mo.plain(model.summary().as_text()),
    ])


def build_assumptions_tab(mo, analysis_df, model, x_name, y_name):
    """Build the Assumptions tab with diagnostic plots.
    
    Args:
        mo: marimo module.
        analysis_df: DataFrame with selected analysis data.
        model: Fitted statsmodels OLS model (or None if validation failed).
        x_name: Name of independent variable.
        y_name: Name of dependent variable.
        
    Returns:
        marimo UI component for assumptions tab.
    """
    if model is None:
        return mo.vstack([
            mo.md("### Assumptions Checks"),
            mo.md(
                "Visualizations will appear after a valid model is estimated."
            ),
        ])
    
    residuals = np.asarray(model.resid)
    fitted_values = np.asarray(model.fittedvalues)
    is_numeric_x = pd.api.types.is_numeric_dtype(analysis_df[x_name])
    
    # Scatter plot with fitted line (only for numeric X)
    if is_numeric_x:
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
            + model.params["_x_numeric"] * regression_x
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
        
        scatter_plot = mo.ui.plotly(scatter_fig)
    else:
        scatter_plot = mo.md(
            f"*Scatter plot not shown for categorical predictor `{x_name}`. "
            "See coefficients table for group means.*"
        )
    
    # Residuals vs fitted values
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
    
    # Normal Q-Q plot
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
    
    return mo.vstack([
        mo.md("### Scatter Plot"),
        scatter_plot,
        mo.md("### Residuals vs Fitted Values"),
        mo.ui.plotly(residual_fig),
        mo.md("### Normal Q-Q Plot"),
        mo.ui.plotly(qq_fig),
    ])

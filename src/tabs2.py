"""Tab content builders for the Linear Regression Explorer."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def build_inspect_tab(mo, data, scatter_plot, x_name, y_name):
    """Build the Inspect the Data tab content."""

    if data.empty:
        return mo.callout(
            "No valid observations are available for the selected variables.",
            kind="warn",
        )

    return mo.vstack([
        mo.md("""
        ## 01. Inspect the data

        In a linear regression, we examine the relationship
        between an independent variable (X) and a dependent
        variable (Y).

        The goal is to find a straight line that best describes
        their relationship.
        """),
        mo.ui.table(data.head(10)),
        mo.md("""
        Before estimating the regression, use a scatter plot
        to explore the relationship between your variables.
        """),
        scatter_plot(data, x=x_name, y=y_name),
    ])
"""Tab content builders for the Linear Regression Explorer."""

import numpy as np
import pandas as pd
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



def build_inspect_tab(mo, data, scatter_plot, x_name, y_name):
    """Build the Inspect the Data tab content."""

    if data.empty:
        return mo.callout(
            "No valid observations are available for the selected variables.",
            kind="warn",
        )

    return mo.vstack([
        mo.md("""
        ## 01 Inspect the data

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
        mo.md("""
        Finally, some BS.
        """),
        mo.ui.table(data[[x_name, y_name]].describe().T),
    ])


def build_linear_tab(mo, data, scatter_plot, x_name, y_name):
    """Build the Linear Association tab content."""

    if data.empty:
        return mo.callout(
            "No valid observations are available for the selected variables.",
            kind="warn",
        )

    # Calculate the correlation before building the layout
    correlation = data[[x_name, y_name]].corr().iloc[0, 1]

    return mo.vstack([
        mo.md("""
        ## 02 Linear Association

        In a linear regression, we examine the relationship between an
        independent variable (X) and a dependent variable (Y). Although
        regression models can include several independent variables to
        explain or predict Y, we will start with the simplest case:
        a model with just one X and one Y.

        The goal of a linear regression is to find a straight line that
        best describes the relationship between X and Y in our data.

        Before estimating the regression, we can use a scatter plot to
        explore the relationship between our chosen variables.
        Each point represents an observation in the dataset.

        Let's take a look at the scatter plot for your selected variables.
        """),

        scatter_plot(data, x=x_name, y=y_name),

        mo.md("""
        What do you observe in the scatter plot? Does there appear to be
        a linear association between X and Y?

        - A positive correlation means higher values of X tend to be
          associated with higher values of Y.
        - A negative correlation means higher values of X tend to be
          associated with lower values of Y.
        - A correlation close to zero suggests little or no linear
          association, although a nonlinear relationship may still exist.

        Use the scatter plot, summary statistics, and correlation coefficient (if available) 
        to describe the relationship between your chosen variables. What do you think the data 
        tell us about the association between X and Y?

        """),

        mo.md(
            f"**Pearson correlation:** `{correlation:.4f}`"
        ),
    ])
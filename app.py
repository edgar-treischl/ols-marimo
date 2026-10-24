# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.3.0",
#     "marimo>=0.25.2",
#     "numpy==2.5.4",
#     "pandas==3.1.0",
#     "plotly==7.1.0",
#     "scipy==1.18.1",
#     "seaborn==0.13.2",
#     "statsmodels==0.15.0",
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
    mo.md(r"""
    Learn how linear regression can help you understand the relationship between two variables. In this notebook, you will choose a dataset, select an independent variable (X) and a dependent variable (Y), and explore the results of a regression analysis — without needing any programming skills.

    For example, you could use the Catholic dataset to investigate whether family income (faminc8, X) is associated with children's reading achievement (read12, Y).

    The notebook runs the necessary calculations automatically and presents the results for you to explore and interpret. You do not need to write, understand, or run any code yourself. The focus is on understanding the basic principles of linear regression and learning what the results tell you about the relationship between two variables.

    > Get started

    Go to the next section and select a dataset, an independent variable (X), and a dependent variable (Y). Then explore the regression results to see what they reveal about your chosen variables.
    """)
    return


@app.cell(hide_code=True)
def import_data():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import statsmodels.api as sm
    from plotly.subplots import make_subplots
    from scipy import stats
    from src.helpers import (
        extract_numeric_columns,
        fit_regression_model,
        load_penguins_data,
        prepare_analysis_data,
        validate_regression_inputs,
    )

    from src.tabs import (
        build_summary_tab,
        build_inspect_tab,
    )

    from src.plots import (
        scatter_plot,
        scatter_regression_plot,
        residual_diagnostics_plot,
        total_variance_plot,
        explained_variance_plot,
        error_plot,
        variance_bar_plot,
    )




    penguins_raw = load_penguins_data()
    penguins = penguins_raw.dropna().copy()
    return (
        build_inspect_tab,
        build_summary_tab,
        error_plot,
        explained_variance_plot,
        fit_regression_model,
        mo,
        np,
        penguins,
        residual_diagnostics_plot,
        scatter_plot,
        scatter_regression_plot,
        sm,
        stats,
        total_variance_plot,
        variance_bar_plot,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 00 Start

    Pick an outcome (**y**) and a predictor variable (**x**) from the *penguins* data.
    """)
    return


@app.cell
def outcome(mo, penguins):
    ## Select the Outcome variable
    numerical_variables = penguins.select_dtypes(include="number").columns.tolist()

    outcome = mo.ui.dropdown(
        options=numerical_variables,
        label="Variable: ",
        value="body_mass_g"
    )

    mo.vstack([
        mo.md("### Outcome variable (Y)"),
        outcome,
    ])
    return (outcome,)


@app.cell
def predictor(mo, outcome, penguins):
    # Select Predictor

    predictor_options = [
        col for col in penguins.columns if col != outcome.value
    ]

    predictor = mo.ui.dropdown(
        options=predictor_options,
        label="Variable: ",
        value=predictor_options[3],
    )

    mo.vstack([
        mo.md("### Predictor variable (X)"),
        predictor,
    ])
    return (predictor,)


@app.cell
def _(build_inspect_tab, mo, outcome, penguins, predictor, scatter_plot):
    # Descriptive statistics
    x_name = predictor.value
    y_name = outcome.value

    inspect_content = build_inspect_tab(
            mo=mo,
            data=penguins,
            scatter_plot=scatter_plot,
            x_name=x_name,
            y_name=y_name,
        )

    tabs = mo.ui.tabs({
        "Inspect the data": inspect_content,
        "Linear Effects": mo.md("Content")
    })
    tabs
    return x_name, y_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 Linear Association

    In a linear regression, we examine the relationship between an independent variable (X) and a dependent variable (Y). Although regression models can include several independent variables to explain or predict Y, we will start with the simplest case: a model with just one X and one Y. The basic principles are the same in more complex models.

    The goal of a linear regression is to find a straight line that best describes the relationship between X and Y in our data. This line helps us understand how changes in X are associated with changes in Y.

    Before estimating the regression, we can use a scatter plot to explore the relationship between our chosen variables. Each point represents an observation in the dataset. By looking at the pattern of points, we can assess whether a linear relationship seems plausible.

    Let's take a look at the scatter plot for your selected variables.
    """)
    return


@app.cell
def _(penguins, scatter_plot, x_name, y_name):
    scatter_plot(penguins, x=x_name, y=y_name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What do you observe in the scatter plot? Does there appear to be a linear association between X and Y? In other words, do the points follow an upward or downward trend that could be described by a straight line?

    To investigate further, you can also examine the summary statistics of your variables. If both X and Y are numerical, look at their correlation coefficient to assess the direction and strength of their linear relationship.

    - A positive correlation means that higher values of X tend to be associated with higher values of Y.
    - A negative correlation means that higher values of X tend to be associated with lower values of Y.
    - A correlation close to zero suggests little or no linear association, although a different type of relationship may still exist.

    Use the scatter plot, summary statistics, and correlation coefficient (if available) to describe the relationship between your chosen variables. What do you think the data tell us about the association between X and Y?
    """)
    return


@app.cell
def _(build_summary_tab, mo, penguins, x_name, y_name):
    build_summary_tab(mo, penguins, x_name, y_name)
    return


@app.cell
def _(penguins, x_name, y_name):
    # Pearson correlation
    correlation = penguins[[x_name, y_name]].corr().iloc[0, 1]
    print(f"Pearson correlation: {correlation:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 The Model

    Regardless of whether the scatter plot shows a clear linear association, we can run a regression to illustrate how the method works and how to interpret its results.

    The statistical software automatically estimates the regression model using your selected X and Y variables. You do not need to write or run any code yourself.

    Below, you will see the regression output. In the next step, we will explore what the results tell us about the relationship between X and Y and how to interpret the key statistics.
    """)
    return


@app.cell
def _(fit_regression_model, penguins, x_name, y_name):
    # Fit the regression model
    model = fit_regression_model(penguins, x_name, y_name)

    # Print model summary
    print(model.summary())
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. The effect of X on Y (coefficient)**

    Look at the coefficient for your independent variable X. It tells you how much Y is expected to change, on average, when X increases by one unit.

    - A **positive coefficient** means that higher values of X are associated with higher values of Y.
    - A **negative coefficient** means that higher values of X are associated with lower values of Y.
    - A coefficient **close to zero** indicates a small estimated linear relationship.

    The size of the coefficient matters too. Its interpretation depends on the units in which X and Y are measured.

    **2. Is the relationship statistically significant? (p-value)**

    The p-value helps us assess whether the observed relationship provides evidence against the hypothesis that the true coefficient of X is zero.

    - If **p < 0.05**, the relationship is conventionally considered statistically significant at the 5% level.
    - If **p ≥ 0.05**, we do not have sufficient evidence to reject the hypothesis that the coefficient is zero.

    Statistical significance does not necessarily mean that the effect is large or practically important. A non-significant result does not prove that there is no relationship.

    **3. How well does the model explain Y? (R-squared)**

    R-squared (\(R^2\)) indicates the proportion of the variation in Y that is explained by the linear regression model using X.

    For example, an \(R^2\) of 0.30 means that the model accounts for 30% of the observed variation in Y.

    Higher values indicate that the model explains a larger share of the variation in Y. However, a high \(R^2\) does not automatically mean that the model is appropriate or that X causes changes in Y.

    ### Your turn

    Look at your regression output and consider the following questions:

    1. Is the estimated coefficient of X positive or negative? What does this mean in the context of your variables?
    2. Is the coefficient statistically significant at the 5% level?
    3. What proportion of the variation in Y is explained by the model?
    4. Taken together, what do the results tell you about the relationship between X and Y?

    **Remember:** Regression identifies an association between variables. By itself, it does not establish a causal effect.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modelfit

    Regardless of the statistical significance or size of the estimated effect, one important question remains: How well does X explain the variation in Y?

    A regression model rarely predicts every observation perfectly. The difference between an observed value of Y and the value predicted by the regression line is called the residual (or prediction error). In the output above, this difference is illustrated in red. The larger the residual, the further the observed value lies from the regression line.
    """)
    return


@app.cell
def _(error_plot, penguins, x_name, y_name):
    chart_error = error_plot(
        penguins,
        x_name,
        y_name,
    )
    chart_error
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The residuals help us understand how well the model fits the data. If the observed values lie close to the regression line, the predictions tend to be more accurate. If they are widely scattered around the line, X alone may not explain much of the variation in Y.

    What do you observe in your example? How close are the observed values to the regression line? Based on the scatter plot and the R
    2
     value in your regression output, how well does X explain the variation in your outcome Y?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variance

    You might think, “There will always be some prediction errors — no big deal!” But how do we know whether these errors are small or large? And how well does our model explain the outcome?

    One useful measure is R-squared (R
    2
    ). It tells us what proportion of the variation in Y is explained by the regression model using X. To understand how it works, let's first consider a situation in which X explains nothing about Y.

    1. Imagine that X explains nothing

    What would the regression line look like if X provided no useful information for predicting Y?

    As you can see in the graph above, the line would be flat (horizontal). Regardless of the value of X, the model would predict the same value of Y: the average of the observed Y values.

    In this situation, the model does not explain why some observations have higher Y values than others. The differences between the observed values and their average represent the total variation in Y. The blue lines in the graph illustrate these deviations.

    This gives us a baseline: how much variation remains when we do not use X to explain differences in Y?
    """)
    return


@app.cell
def _(penguins, total_variance_plot, x_name, y_name):
    chart_total_var = total_variance_plot(
        penguins,
        x_name,
        y_name,
    )
    chart_total_var
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. Now let X explain what it can

    Our regression model does better than simply predicting the average for everyone. It uses X to estimate different values of Y, represented by the fitted regression line.

    The next graph illustrates the explained variation in green: the part of the variation in Y that the regression model accounts for by predicting values that differ from the overall average.

    The more the regression model improves on simply predicting the average, the more variation it explains. However, some variation will usually remain unexplained — the observed values will not all fall exactly on the regression line.

    Let's look at the explained variation in the next graph.
    """)
    return


@app.cell
def _(explained_variance_plot, penguins, x_name, y_name):
    chart_explained_var = explained_variance_plot(
        penguins,
        x_name,
        y_name,
    )
    chart_explained_var
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Understanding R-squared

    We now know the total variation in Y and how much of that variation is explained by our regression model. We can use these quantities to calculate R-squared (R
    2
    ), which tells us what proportion of the variation in Y is explained by the model using X.

    R-squared ranges from 0 to 1 (or 0% to 100%) in a standard linear regression with an intercept:

    - R² = 0 means that the model explains none of the variation in Y beyond predicting its average.
    - R² = 0.30 means that the model explains 30% of the variation in Y.
    - R² = 1 means that the model explains all the observed variation in Y.

    The final bar plot brings the three components together: the total variation, the explained variation, and the unexplained variation (the residuals). The explained and unexplained components add up to the total variation.

    Now it's your turn! Look at the bar plot and the R-squared value in your regression output. How much of the variation in your outcome Y does X explain? Would you say that your model explains the outcome well, or does much of the variation remain unexplained?
    """)
    return


@app.cell
def _(penguins, variance_bar_plot, x_name, y_name):
    chart_var_bar = variance_bar_plot(penguins, x_name, y_name)
    chart_var_bar
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 Diagnostics

    1. Residuals vs. Fitted: Are we missing a pattern?

    So far, we have explored how well X explains Y. But how can we check whether our regression model describes the data appropriately?

    The Residuals vs. Fitted plot helps us investigate this question. A residual is the difference between an observed value of Y and the value predicted by our regression model. The fitted values are the predictions made by the model.

    In this plot, the horizontal axis shows the predicted values of Y, while the vertical axis shows the residuals.

    Ideally, the residuals should be scattered randomly around the horizontal line at zero, without a clear pattern. This suggests that the model has captured the main linear relationship in the data.

    Look for the following patterns:

    Random scatter around zero: The linear model may be appropriate for the data.
    A curved pattern: The relationship between X and Y may not be adequately described by a straight line.
    A funnel-shaped pattern: The prediction errors may vary in size across different fitted values.
    Unusual points: Some observations may have much larger residuals than others and deserve closer inspection.

    Your turn: Look at the plot for your chosen variables. Do the residuals appear randomly scattered around zero, or can you identify a pattern? What might this tell you about your regression model?
    """)
    return


@app.cell
def _(penguins, scatter_regression_plot, x_name, y_name):
    # Scatter plot and fitted regression line
    chart = scatter_regression_plot(
        penguins,
        x_name,
        y_name,
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. Residual distribution: How are our prediction errors distributed?

    The Residuals vs. Fitted plot helps us identify patterns in prediction errors. The residual distribution provides a different perspective: it shows how the errors are distributed across all observations.

    If the residuals are concentrated around zero, the model's predictions tend to be close to the observed values. A roughly symmetric, bell-shaped distribution suggests that the errors may be approximately normally distributed.

    However, the distribution can reveal potential issues:

    A roughly symmetric, bell-shaped distribution: The residuals may be approximately normally distributed.
    A skewed distribution: The errors may be more pronounced in one direction than the other.
    Long tails or extreme values: Some observations may have unusually large prediction errors.
    A distribution far from zero: The model may systematically overpredict or underpredict the outcome, although the shape and location of the distribution should be considered separately.

    A normal residual distribution is particularly relevant for certain statistical tests and confidence intervals. It is not, by itself, proof that the regression model is appropriate.

    Your turn: Examine the residual distribution for your chosen variables. Is it roughly symmetric and centered around zero? Are there any signs of skewness or unusually large errors? What do these observations suggest about your model?
    """)
    return


@app.cell
def _(penguins, residual_diagnostics_plot, x_name, y_name):
    # Residual diagnostics
    chart_residual_diagnostics = residual_diagnostics_plot(
        penguins,
        x_name,
        y_name,
    )
    chart_residual_diagnostics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. Checking the regression assumptions: Normality and heteroscedasticity

    We have explored the regression results and inspected the residual plots. We can now use statistical tests to investigate two additional aspects of our model: normality and heteroscedasticity.

    These tests help us assess whether certain assumptions of the linear regression model are reasonable. They do not tell us whether the model is automatically good or bad, but they can highlight potential problems that affect how we interpret the results.

    Normality test: Are the residuals approximately normally distributed?

    For some statistical tests and confidence intervals in linear regression, we assume that the residuals are approximately normally distributed.

    The normality test helps us assess whether the distribution of residuals differs significantly from a normal distribution.

    p-value greater than 0.05: We do not have sufficient evidence to reject the assumption of normality.
    p-value of 0.05 or less: We have evidence that the residual distribution differs from a normal distribution.

    Keep in mind that a non-significant result does not prove normality. The test can also be sensitive to sample size, so it is useful to consider the residual distribution plot alongside the test result.

    Heteroscedasticity test: Do the errors vary in size?

    Another important consideration is whether the variance of the residuals remains approximately constant across different values of X or the fitted values of Y. This property is called homoscedasticity.

    If the variance changes across observations, we call this heteroscedasticity.

    p-value greater than 0.05: We do not have sufficient evidence to detect heteroscedasticity.
    p-value of 0.05 or less: We have evidence of heteroscedasticity, meaning that the variance of the errors may not be constant.

    Heteroscedasticity does not necessarily make the estimated regression coefficients biased, but it can make conventional standard errors, p-values, and confidence intervals unreliable. Robust standard errors can often help address this issue.
    """)
    return


@app.cell
def _(model, np, sm, stats):
    # Normality and heteroscedasticity tests
    alpha = 0.05
    fitted_values = np.asarray(model.fittedvalues)
    residuals = np.asarray(model.resid)

    if 3 <= len(residuals) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)

        print("Shapiro-Wilk normality test")
        print(f"Statistic: {shapiro_stat:.4f}")
        print(f"P-value: {shapiro_p:.4g}")

        if shapiro_p < alpha:
            print(
                "Interpretation: Reject the null hypothesis. "
                "There is evidence that the residuals are not "
                "normally distributed."
            )
        else:
            print(
                "Interpretation: Fail to reject the null hypothesis. "
                "There is insufficient evidence to conclude that "
                "the residuals deviate from normality."
            )
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

    if bp_pvalue < alpha:
        print(
            "Interpretation: Reject the null hypothesis. "
            "There is evidence of heteroscedasticity "
            "(non-constant error variance)."
        )
    else:
        print(
            "Interpretation: Fail to reject the null hypothesis. "
            "There is insufficient evidence of heteroscedasticity; "
            "the constant-variance assumption is not rejected."
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Your turn: What do the tests tell you?

    Look at the results of both tests for your chosen variables.

    Does the normality test provide evidence that the residuals are not normally distributed?
    Does the heteroscedasticity test provide evidence that the error variance is not constant?
    Considering these results alongside the residual plots, what can you conclude about the assumptions of your regression model?

    Remember: Statistical tests provide evidence, not absolute proof. Interpret their results in combination with the plots, the sample size, and your understanding of the data.
    """)
    return


if __name__ == "__main__":
    app.run()

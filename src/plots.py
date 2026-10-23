
import numpy as np
import altair as alt
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
)
import statsmodels.api as sm

def scatter_plot(data, x, y, title=None):
    """
    Plot y against x, automatically choosing:
      - Numerical x: scatter plot + OLS line + 95% mean CI
      - Categorical x: jittered strip plot + box plot

    Missing x/y observations are removed.
    """
    df = data[[x, y]].dropna().copy()

    if df.empty:
        raise ValueError("No complete observations available.")

    if not is_numeric_dtype(df[y]):
        raise TypeError(f"y='{y}' must be numeric.")

    if is_datetime64_any_dtype(df[x]):
        raise TypeError(
            "Datetime x is not supported by this function yet."
        )

    # Shared styling
    chart_title = title or f"{y} vs. {x}"

    tooltip = [
        alt.Tooltip(field=x, title=x),
        alt.Tooltip(field=y, type="quantitative", title=y),
    ]

    # --------------------------------------------------
    # Numerical x: scatter + OLS + confidence interval
    # --------------------------------------------------
    if is_numeric_dtype(df[x]):
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df[x].nunique() < 2:
            raise ValueError(
                f"x='{x}' must contain at least two distinct values."
            )

        model = sm.OLS(
            df[y],
            sm.add_constant(df[[x]], has_constant="add"),
        ).fit()

        x_grid = np.linspace(df[x].min(), df[x].max(), 200)
        grid = pd.DataFrame({x: x_grid})

        predictions = model.get_prediction(
            sm.add_constant(grid[[x]], has_constant="add")
        ).summary_frame(alpha=0.05)

        band_data = pd.DataFrame({
            x: x_grid,
            "fit": predictions["mean"].to_numpy(),
            "lower": predictions["mean_ci_lower"].to_numpy(),
            "upper": predictions["mean_ci_upper"].to_numpy(),
        })

        band = alt.Chart(band_data).mark_area(
            color="#4C78A8", opacity=0.20
        ).encode(
            x=alt.X(
                f"{x}:Q",
                title=x,
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("lower:Q", title=y),
            y2="upper:Q",
        )

        line = alt.Chart(band_data).mark_line(
            color="#2B5C8A", strokeWidth=2
        ).encode(
            x=alt.X(f"{x}:Q"),
            y=alt.Y("fit:Q"),
        )

        points = alt.Chart(df).mark_circle(
            size=55, opacity=0.65
        ).encode(
            x=alt.X(
                f"{x}:Q",
                title=x,
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                f"{y}:Q",
                title=y,
                scale=alt.Scale(zero=False),
            ),
            tooltip=tooltip,
        )

        chart = band + line + points

    # --------------------------------------------------
    # Categorical x: jittered strip plot + box plot
    # --------------------------------------------------
    else:
        # Preserve the existing order of categorical values.
        categories = df[x].drop_duplicates().tolist()

        df[x] = pd.Categorical(
            df[x],
            categories=categories,
            ordered=True,
        )

        points = (
            alt.Chart(df)
            .transform_calculate(
                jitter="(random() - 0.5) * 0.7"
            )
            .mark_circle(size=55, opacity=0.55)
            .encode(
                x=alt.X(
                    f"{x}:N",
                    title=x,
                    sort=categories,
                ),
                xOffset=alt.XOffset("jitter:Q"),
                y=alt.Y(
                    f"{y}:Q",
                    title=y,
                    scale=alt.Scale(zero=False),
                ),
                tooltip=tooltip,
            )
        )

        boxes = alt.Chart(df).mark_boxplot(
            extent=1.5,
            size=35,
            opacity=0.5,
            color="#4C78A8",
        ).encode(
            x=alt.X(
                f"{x}:N",
                title=x,
                sort=categories,
            ),
            y=alt.Y(
                f"{y}:Q",
                title=y,
                scale=alt.Scale(zero=False),
            ),
        )

        chart = boxes + points

    return (
        chart
        .properties(
            title=chart_title,
            width="container",
            height=400,
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            grid=True,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
    )





import numpy as np
import pandas as pd
import altair as alt
from pandas.api.types import is_numeric_dtype


def total_variance_plot(data, x, y, title=None):
    """
    Plot total variation in Y around its mean.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    x : str
        Column name for the independent variable.
    y : str
        Column name for the numerical dependent variable.
    title : str, optional
        Chart title.

    Returns
    -------
    altair.Chart
        Scatter plot with deviations from the mean of Y.
    """
    df = data[[x, y]].dropna().copy()

    # Validate data
    if df.empty:
        raise ValueError("No complete observations available.")

    if not is_numeric_dtype(df[x]):
        raise TypeError(f"x='{x}' must be numeric.")

    if not is_numeric_dtype(df[y]):
        raise TypeError(f"y='{y}' must be numeric.")

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        raise ValueError("No valid finite observations available.")

    # Mean of Y
    mean_y = df[y].mean()

    # Chart title and tooltips
    chart_title = title or "Total Variance"

    tooltip = [
        alt.Tooltip(field=x, type="quantitative", title=x),
        alt.Tooltip(field=y, type="quantitative", title=y),
        alt.Tooltip(
            field="deviation",
            type="quantitative",
            title="Deviation from mean",
            format=".3f",
        ),
    ]

    # Calculate deviations from the mean
    df["deviation"] = df[y] - mean_y

    # Vertical segments: observed Y to mean of Y
    segments = alt.Chart(df).mark_rule(
        color="#0571b0",
        strokeWidth=1.5,
        opacity=0.7,
    ).encode(
        x=alt.X(
            f"{x}:Q",
            title=x,
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            f"{y}:Q",
            title=y,
            scale=alt.Scale(zero=False),
        ),
        y2=alt.Y2(datum=mean_y),
        tooltip=tooltip,
    )

    # Observed data points
    points = alt.Chart(df).mark_circle(
        size=55,
        opacity=0.75,
        color="#4C78A8",
    ).encode(
        x=alt.X(
            f"{x}:Q",
            title=x,
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            f"{y}:Q",
            title=y,
            scale=alt.Scale(zero=False),
        ),
        tooltip=tooltip,
    )

    # Horizontal line representing the mean of Y
    mean_line = alt.Chart(
        pd.DataFrame({"mean_y": [mean_y]})
    ).mark_rule(
        color="#D62728",
        strokeDash=[6, 4],
        strokeWidth=2,
    ).encode(
        y=alt.Y("mean_y:Q"),
        tooltip=[
            alt.Tooltip(
                "mean_y:Q",
                title=f"Mean of {y}",
                format=".3f",
            )
        ],
    )

    return (
        (segments + mean_line + points)
        .properties(
            title=chart_title,
            width="container",
            height=400,
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            grid=True,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
    )




import numpy as np
import pandas as pd
import altair as alt
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype


def explained_variance_plot(data, x, y, title=None):
    """
    Plot the variation in Y explained by a linear regression on X.

    Green segments show the difference between fitted values
    and the mean of Y.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    x : str
        Column name for the independent variable.
    y : str
        Column name for the dependent variable.
    title : str, optional
        Chart title.

    Returns
    -------
    altair.Chart
        Scatter plot with fitted regression line and explained
        deviations from the mean of Y.
    """
    df = data[[x, y]].dropna().copy()

    # Validate data
    if df.empty:
        raise ValueError("No complete observations available.")

    if not is_numeric_dtype(df[x]):
        raise TypeError(f"x='{x}' must be numeric.")

    if not is_numeric_dtype(df[y]):
        raise TypeError(f"y='{y}' must be numeric.")

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        raise ValueError("No valid finite observations available.")

    if df[x].nunique() < 2:
        raise ValueError(
            f"x='{x}' must contain at least two distinct values."
        )

    # Fit OLS regression model
    model = sm.OLS(
        df[y],
        sm.add_constant(df[[x]], has_constant="add"),
    ).fit()

    # Predicted values and mean of Y
    df["ypred"] = model.predict(
        sm.add_constant(df[[x]], has_constant="add")
    )
    mean_y = df[y].mean()

    # Chart title and tooltips
    chart_title = title or "Variance explained by X"

    tooltip = [
        alt.Tooltip(field=x, type="quantitative", title=x),
        alt.Tooltip(field=y, type="quantitative", title=y),
        alt.Tooltip(
            field="ypred",
            type="quantitative",
            title="Predicted Y",
            format=".3f",
        ),
    ]

    # Green segments: fitted values to the mean of Y
    segments = alt.Chart(df).mark_rule(
        color="#008837",
        strokeWidth=1.5,
        opacity=0.75,
    ).encode(
        x=alt.X(
            f"{x}:Q",
            title=x,
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            "ypred:Q",
            title=y,
            scale=alt.Scale(zero=False),
        ),
        y2=alt.Y2(datum=mean_y),
        tooltip=tooltip,
    )

    # Regression line
    x_grid = np.linspace(df[x].min(), df[x].max(), 200)
    grid = pd.DataFrame({x: x_grid})

    grid["ypred"] = model.predict(
        sm.add_constant(grid[[x]], has_constant="add")
    )

    line = alt.Chart(grid).mark_line(
        color="black",
        strokeWidth=2,
    ).encode(
        x=alt.X(f"{x}:Q", title=x),
        y=alt.Y("ypred:Q", title=y),
    )

    # Horizontal line representing the mean of Y
    mean_line = alt.Chart(
        pd.DataFrame({"mean_y": [mean_y]})
    ).mark_rule(
        color="#D62728",
        strokeDash=[6, 4],
        strokeWidth=2,
    ).encode(
        y=alt.Y("mean_y:Q"),
        tooltip=[
            alt.Tooltip(
                "mean_y:Q",
                title=f"Mean of {y}",
                format=".3f",
            )
        ],
    )

    # Invisible points preserve the original R plot's structure;
    # tooltips remain available on the green segments.
    return (
        (segments + mean_line + line)
        .properties(
            title=chart_title,
            width="container",
            height=400,
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            grid=True,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
    )



import numpy as np
import pandas as pd
import altair as alt
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype


def error_plot(data, x, y, title=None):
    """
    Plot observed values, the fitted OLS regression line,
    and residuals (observed Y - predicted Y).

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    x : str
        Column name for the independent variable.
    y : str
        Column name for the dependent variable.
    title : str, optional
        Chart title.

    Returns
    -------
    altair.Chart
        Scatter plot with regression line and residual segments.
    """
    df = data[[x, y]].dropna().copy()

    # Validate data
    if df.empty:
        raise ValueError("No complete observations available.")

    if not is_numeric_dtype(df[x]):
        raise TypeError(f"x='{x}' must be numeric.")

    if not is_numeric_dtype(df[y]):
        raise TypeError(f"y='{y}' must be numeric.")

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        raise ValueError("No valid finite observations available.")

    if df[x].nunique() < 2:
        raise ValueError(
            f"x='{x}' must contain at least two distinct values."
        )

    # Fit OLS regression
    X = sm.add_constant(df[[x]], has_constant="add")
    model = sm.OLS(df[y], X).fit()

    # Calculate fitted values and residuals
    df["ypred"] = model.predict(X)
    df["residual"] = df[y] - df["ypred"]

    # Generate points along the regression line
    x_grid = np.linspace(df[x].min(), df[x].max(), 200)
    grid = pd.DataFrame({x: x_grid})
    grid["ypred"] = model.predict(
        sm.add_constant(grid[[x]], has_constant="add")
    )

    chart_title = title or "Total error"

    tooltip = [
        alt.Tooltip(field=x, type="quantitative", title=x),
        alt.Tooltip(field=y, type="quantitative", title=y),
        alt.Tooltip(
            field="ypred",
            type="quantitative",
            title="Predicted Y",
            format=".3f",
        ),
        alt.Tooltip(
            field="residual",
            type="quantitative",
            title="Residual",
            format=".3f",
        ),
    ]

    # Red segments: observed Y to predicted Y
    segments = alt.Chart(df).mark_rule(
        color="#ca0020",
        strokeWidth=1.5,
        opacity=0.75,
    ).encode(
        x=alt.X(
            f"{x}:Q",
            title=x,
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            f"{y}:Q",
            title=y,
            scale=alt.Scale(zero=False),
        ),
        y2=alt.Y2("ypred:Q"),
        tooltip=tooltip,
    )

    # Observed data points
    points = alt.Chart(df).mark_circle(
        size=65,
        opacity=0.8,
        color="#4C78A8",
    ).encode(
        x=alt.X(
            f"{x}:Q",
            title=x,
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            f"{y}:Q",
            title=y,
            scale=alt.Scale(zero=False),
        ),
        tooltip=tooltip,
    )

    # Fitted regression line
    line = alt.Chart(grid).mark_line(
        color="black",
        strokeWidth=2,
    ).encode(
        x=alt.X(f"{x}:Q", title=x),
        y=alt.Y("ypred:Q", title=y),
    )

    return (
        (segments + line + points)
        .properties(
            title=chart_title,
            width="container",
            height=400,
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            grid=True,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
    )




import numpy as np
import pandas as pd
import statsmodels.api as sm
import altair as alt


def variance_bar_plot(data, x, y):
    # Prepare data
    df = data[[x, y]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()

    if df.empty:
        raise ValueError("No complete observations available.")

    if df[x].nunique() < 2:
        raise ValueError(f"x='{x}' must have at least two distinct values.")

    # Fit linear regression
    X = sm.add_constant(df[[x]], has_constant="add")
    model = sm.OLS(df[y], X).fit()

    observed = df[y]
    predicted = model.predict(X)
    mean_y = observed.mean()

    # Calculate sums of squares
    total_ss = ((observed - mean_y) ** 2).sum()
    explained_ss = ((predicted - mean_y) ** 2).sum()
    residual_ss = ((observed - predicted) ** 2).sum()

    if total_ss == 0:
        raise ValueError("Y has no variation; R-squared is undefined.")

    # Convert to percentages of total variation
    ssdata = pd.DataFrame({
        "SS": ["Total", "Explained", "Unexplained"],
        "value": [
            100,
            100 * explained_ss / total_ss,
            100 * residual_ss / total_ss,
        ],
    })

    # Plot variance components
    colors = ["#0571b0", "#008837", "#ca0020"]

    bars = alt.Chart(ssdata).mark_bar().encode(
        x=alt.X(
            "SS:N",
            title="",
            sort=["Total", "Explained", "Unexplained"],
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            "value:Q",
            title="% of total variation",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "SS:N",
            scale=alt.Scale(
                domain=["Total", "Explained", "Unexplained"],
                range=colors,
            ),
            legend=alt.Legend(title="Variance component"),
        ),
        tooltip=[
            alt.Tooltip("SS:N", title="Component"),
            alt.Tooltip("value:Q", title="Percentage", format=".2f"),
        ],
    )

    return (
        bars.properties(
            title="Variance components",
            width="container",
            height=400,
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            grid=True,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
        .configure_legend(
            labelFontSize=13,
            titleFontSize=13,
        )
    )



import numpy as np
import pandas as pd
import altair as alt
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype


def scatter_regression_plot(data, x, y, title=None):
    """
    Scatter plot with OLS regression line and 95% confidence
    interval for the mean prediction.
    """
    df = data[[x, y]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()

    if df.empty:
        raise ValueError("No complete observations available.")

    if not is_numeric_dtype(df[x]):
        raise TypeError(f"x='{x}' must be numeric.")

    if not is_numeric_dtype(df[y]):
        raise TypeError(f"y='{y}' must be numeric.")

    if df[x].nunique() < 2:
        raise ValueError(
            f"x='{x}' must contain at least two distinct values."
        )

    # Fit OLS regression
    X = sm.add_constant(df[[x]], has_constant="add")
    model = sm.OLS(df[y], X).fit()

    # Generate predictions across the range of X
    x_grid = np.linspace(df[x].min(), df[x].max(), 200)
    grid = pd.DataFrame({x: x_grid})

    predictions = model.get_prediction(
        sm.add_constant(grid[[x]], has_constant="add")
    ).summary_frame(alpha=0.05)

    band_data = pd.DataFrame({
        x: x_grid,
        "fit": predictions["mean"].to_numpy(),
        "lower": predictions["mean_ci_lower"].to_numpy(),
        "upper": predictions["mean_ci_upper"].to_numpy(),
    })

    chart_title = title or f"{y} vs. {x}"

    tooltip = [
        alt.Tooltip(field=x, type="quantitative", title=x),
        alt.Tooltip(field=y, type="quantitative", title=y),
    ]

    # 95% confidence interval for the mean prediction
    band = alt.Chart(band_data).mark_area(
        color="#4C78A8",
        opacity=0.20,
    ).encode(
        x=alt.X(
            f"{x}:Q",
            title=x,
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y("lower:Q", title=y),
        y2="upper:Q",
    )

    # OLS regression line
    line = alt.Chart(band_data).mark_line(
        color="black",
        strokeWidth=2,
    ).encode(
        x=alt.X(f"{x}:Q", title=x),
        y=alt.Y("fit:Q", title=y),
    )

    # Observed data points
    points = alt.Chart(df).mark_circle(
        size=55,
        opacity=0.65,
        color="#4C78A8",
    ).encode(
        x=alt.X(
            f"{x}:Q",
            title=x,
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            f"{y}:Q",
            title=y,
            scale=alt.Scale(zero=False),
        ),
        tooltip=tooltip,
    )

    return (
        (band + line + points)
        .properties(
            title=chart_title,
            width="container",
            height=400,
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            grid=True,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
    )




import numpy as np
import pandas as pd
import altair as alt
import statsmodels.api as sm


def residual_diagnostics_plot(data, x, y):
    """
    Display residuals vs. fitted values and the residual distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    x : str
        Independent variable column name.
    y : str
        Dependent variable column name.

    Returns
    -------
    altair.HConcatChart
        Two-panel regression diagnostic plot.
    """
    # Prepare data
    df = data[[x, y]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()

    if df.empty:
        raise ValueError("No complete observations available.")

    if not pd.api.types.is_numeric_dtype(df[x]):
        raise TypeError(f"x='{x}' must be numeric.")

    if not pd.api.types.is_numeric_dtype(df[y]):
        raise TypeError(f"y='{y}' must be numeric.")

    if df[x].nunique() < 2:
        raise ValueError(
            f"x='{x}' must contain at least two distinct values."
        )

    # Fit OLS regression
    X = sm.add_constant(df[[x]], has_constant="add")
    model = sm.OLS(df[y], X).fit()

    # Calculate fitted values and residuals
    df["fitted"] = np.asarray(model.fittedvalues)
    df["residual"] = np.asarray(model.resid)

    # -----------------------------------------------
    # 1. Residuals vs. Fitted
    # -----------------------------------------------
    points = alt.Chart(df).mark_circle(
        size=55,
        opacity=0.65,
        color="#4C78A8",
    ).encode(
        x=alt.X(
            "fitted:Q",
            title="Fitted values",
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            "residual:Q",
            title="Residuals",
            scale=alt.Scale(zero=False),
        ),
        tooltip=[
            alt.Tooltip(
                "fitted:Q",
                title="Fitted",
                format=".3f",
            ),
            alt.Tooltip(
                "residual:Q",
                title="Residual",
                format=".3f",
            ),
        ],
    )

    zero_line = alt.Chart(
        pd.DataFrame({"zero": [0]})
    ).mark_rule(
        color="red",
        strokeDash=[6, 4],
        strokeWidth=2,
    ).encode(
        y=alt.Y("zero:Q"),
    )

    residuals_vs_fitted = (
        points + zero_line
    ).properties(
        title="Residuals vs. Fitted",
        height=400,
    )

    # -----------------------------------------------
    # 2. Residual distribution
    # -----------------------------------------------
    histogram = alt.Chart(df).mark_bar(
        color="#4C78A8",
        opacity=0.8,
    ).encode(
        x=alt.X(
            "residual:Q",
            bin=alt.Bin(maxbins=30),
            title="Residuals",
        ),
        y=alt.Y(
            "count():Q",
            title="Frequency",
        ),
        tooltip=[
            alt.Tooltip(
                "count():Q",
                title="Frequency",
            ),
        ],
    ).properties(
        title="Residual distribution",
        height=400,
    )

    # -----------------------------------------------
    # Combine panels
    # -----------------------------------------------
    return (
        alt.hconcat(
            residuals_vs_fitted,
            histogram,
            spacing=25,
        )
        .resolve_scale(
            x="independent",
            y="independent",
        )
        .properties(
            title="Regression diagnostics",
            autosize=alt.AutoSizeParams(
                type="fit",
                contains="padding",
            ),
        )
        .configure_view(
            continuousWidth=400,
            continuousHeight=400,
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            grid=True,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
    )


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
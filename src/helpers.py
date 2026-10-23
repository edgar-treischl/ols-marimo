"""Helper functions for data loading, preparation, and analysis."""

from io import BytesIO
from urllib.request import urlopen
import pandas as pd
import statsmodels.api as sm
import numpy as np
import altair as alt
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
)


def load_penguins_data(data_path):
    """Load penguin dataset from CSV file or URL.
    
    Args:
        data_path: Path to the penguins.csv file (supports http/https URLs).
        
    Returns:
        DataFrame containing the penguins dataset.
    """
    if str(data_path).startswith(("http://", "https://")):
        with urlopen(str(data_path)) as response:
            return pd.read_csv(BytesIO(response.read()))
    return pd.read_csv(str(data_path))


def extract_numeric_columns(df):
    """Extract numeric columns from a DataFrame.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        List of numeric column names.
        
    Raises:
        ValueError: If no numeric columns are found.
    """
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise ValueError("The dataset contains no numeric columns.")
    return numeric_columns




def prepare_analysis_data(df, x_name, y_name):
    """Prepare numeric or categorical X and numeric Y for regression."""
    data = (
        df[[x_name, y_name]]
        .replace([float("inf"), -float("inf")], float("nan"))
        .dropna()
        .copy()
    )

    if pd.api.types.is_numeric_dtype(data[x_name]):
        data["_x_numeric"] = pd.to_numeric(data[x_name])
        data.attrs["predictor_columns"] = ["_x_numeric"]
    else:
        dummies = pd.get_dummies(
            data[x_name],
            prefix=x_name,
            prefix_sep="__",
            drop_first=True,
            dtype=float,
        )
        predictor_columns = list(dummies.columns)
        data = pd.concat([data, dummies], axis=1)
        data.attrs["predictor_columns"] = predictor_columns

    return data


def fit_regression_model(analysis_df, x_name, y_name):
    """Fit OLS using numeric or dummy-encoded predictors."""
    predictor_columns = analysis_df.attrs.get("predictor_columns")

    if not predictor_columns:
        if pd.api.types.is_numeric_dtype(analysis_df[x_name]):
            predictor_columns = [x_name]
        else:
            predictor_columns = [
                c for c in analysis_df.columns
                if c.startswith(f"{x_name}__")
            ]

    if not predictor_columns:
        raise ValueError("No usable predictor columns for X.")

    X = analysis_df[predictor_columns].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = analysis_df[y_name].astype(float)

    return sm.OLS(y, X).fit()


def validate_regression_inputs(analysis_df, x_name, y_name):
    """Validate regression inputs."""
    if x_name == y_name:
        return False, "Please select different variables for X and Y."

    if analysis_df.empty:
        return False, "No valid observations are available."

    if len(analysis_df) < 3:
        return False, "At least three valid observations are required."

    if not pd.api.types.is_numeric_dtype(analysis_df[y_name]):
        return False, "Y must be numeric."

    if analysis_df[y_name].nunique() < 2:
        return False, "The dependent variable is constant."

    if analysis_df[x_name].nunique() < 2:
        return False, "X must contain at least two distinct values."

    predictor_columns = analysis_df.attrs.get("predictor_columns", [])

    if not predictor_columns:
        if pd.api.types.is_numeric_dtype(analysis_df[x_name]):
            predictor_columns = [x_name]
        else:
            predictor_columns = [
                c for c in analysis_df.columns
                if c.startswith(f"{x_name}__")
            ]

    if not predictor_columns:
        return False, "X has no usable predictor columns after encoding."

    if not analysis_df[predictor_columns].nunique().gt(1).any():
        return False, "X has no variation in its encoded predictors."

    return True, None






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
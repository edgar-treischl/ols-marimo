"""Helper functions for data loading, preparation, and analysis."""

from io import BytesIO
from urllib.request import urlopen
import pandas as pd
import statsmodels.api as sm

import seaborn as sns

def load_penguins_data(data_path=None):
    """Load the Palmer Penguins dataset."""
    if data_path is None:
        return sns.load_dataset("penguins")
    return pd.read_csv(data_path)

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







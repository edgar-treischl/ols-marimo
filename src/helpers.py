"""Helper functions for data loading, preparation, and analysis."""

from io import BytesIO
from urllib.request import urlopen

import pandas as pd
import statsmodels.api as sm


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
    """Prepare data for regression analysis.
    
    Selects specified columns, removes infinite values, and drops NaN rows.
    
    Args:
        df: Input DataFrame.
        x_name: Name of independent variable.
        y_name: Name of dependent variable.
        
    Returns:
        DataFrame with selected columns, cleaned for analysis.
    """
    return (
        df[[x_name, y_name]]
        .replace([float("inf"), -float("inf")], float("nan"))
        .dropna()
        .copy()
    )


def fit_regression_model(analysis_df, x_name, y_name):
    """Fit an OLS regression model.
    
    Args:
        analysis_df: DataFrame with x and y columns.
        x_name: Name of independent variable.
        y_name: Name of dependent variable.
        
    Returns:
        Fitted statsmodels OLS regression model.
    """
    X = sm.add_constant(
        analysis_df[[x_name]],
        has_constant="add",
    )
    y_values = analysis_df[y_name]
    return sm.OLS(y_values, X).fit()


def validate_regression_inputs(analysis_df, x_name, y_name):
    """Validate inputs for regression analysis.
    
    Args:
        analysis_df: DataFrame with x and y columns.
        x_name: Name of independent variable.
        y_name: Name of dependent variable.
        
    Returns:
        Tuple of (is_valid: bool, error_message: str or None).
    """
    if x_name == y_name:
        return False, "Please select different variables for X and Y."
    
    if analysis_df.empty:
        return False, "No valid observations are available for the selected variables."
    
    if len(analysis_df) < 3:
        return False, "At least three valid observations are required."
    
    if analysis_df[x_name].nunique() < 2:
        return False, "The independent variable must contain at least two distinct values."
    
    if analysis_df[y_name].nunique() < 2:
        return False, "The dependent variable is constant; regression statistics are not meaningful."
    
    return True, None

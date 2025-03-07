import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def flatten_columns(df):
    """
    If the DataFrame columns are a MultiIndex, flatten them.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_log_returns(df):
    """
    Calculates log returns for the 'Close' price and adds a new column 'Log_Returns'.
    """
    df = flatten_columns(df)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def clean_log_returns(series):
    """
    Replaces infinite values with NaN and drops them.
    """
    series = series.replace([np.inf, -np.inf], np.nan)
    return series.dropna()

from scipy.stats import skew, kurtosis

def compute_cumulants(series):
    """
    Computes mean, standard deviation, skewness, and excess kurtosis of the series.
    """
    mu = series.mean()
    sigma = series.std()
    skew_val = skew(series)
    kurt_val = kurtosis(series)  # excess kurtosis
    return mu, sigma, skew_val, kurt_val

def add_economic_features(df):
    """
    Adds global economic features based on log returns: mean, std, skewness, and kurtosis.
    """
    mu, sigma, skew_val, kurt_val = compute_cumulants(df['Log_Returns'])
    df['LogReturns_Mean'] = mu
    df['LogReturns_Std'] = sigma
    df['LogReturns_Skew'] = skew_val
    df['LogReturns_Kurtosis'] = kurt_val
    return df

def calculate_var(series, percentile=5):
    """
    Calculates the Value at Risk (VaR) at the given percentile.
    """
    return np.percentile(series, percentile)
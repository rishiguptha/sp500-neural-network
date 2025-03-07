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
    Adds global economic features (cumulants) computed from Log_Returns.
    """
    mu, sigma, skew_val, kurt_val = compute_cumulants(df['Log_Returns'])
    df['LogReturns_Mean'] = mu
    df['LogReturns_Std'] = sigma
    df['LogReturns_Skew'] = skew_val
    df['LogReturns_Kurtosis'] = kurt_val
    return df

def add_rolling_indicators(df, window=14):
    """
    Adds rolling indicators: rolling mean and rolling std for the 'Close' price.
    """
    df[f'RollingMean_{window}'] = df['Close'].rolling(window=window).mean()
    df[f'RollingStd_{window}'] = df['Close'].rolling(window=window).std()
    df = df.dropna()  # Remove rows with incomplete rolling window
    return df

def compute_rsi(series, window=14):
    """
    Computes the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame:
      - SMA (20-day and 50-day)
      - EMA (20-day)
      - RSI (14-day)
    """
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    df['RSI_14'] = compute_rsi(df['Close'], window=14)
    
    # Drop rows with NA values produced by rolling computations
    df = df.dropna()
    return df

def calculate_var(series, percentile=5):
    """
    Calculates the Value at Risk (VaR) at the given percentile.
    """
    return np.percentile(series, percentile)
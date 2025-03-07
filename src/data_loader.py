import pandas as pd
import numpy as np

def compute_technical_indicators(df):
    """
    Compute a set of technical indicators and add them to the dataframe.
    Assumes df has columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
    """

    # --- Simple Moving Averages and Bollinger Bands ---
    sma_windows = [5, 10, 20, 50, 100, 200]
    for window in sma_windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'STD_{window}'] = df['Close'].rolling(window=window).std()
        # Bollinger Bands: Upper and Lower bands (using 2 standard deviations)
        df[f'BB_upper_{window}'] = df[f'SMA_{window}'] + 2 * df[f'STD_{window}']
        df[f'BB_lower_{window}'] = df[f'SMA_{window}'] - 2 * df[f'STD_{window}']
    
    # --- Exponential Moving Averages (for MACD) ---
    ema_short = 12
    ema_long = 26
    df[f'EMA_{ema_short}'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df[f'EMA_{ema_long}'] = df['Close'].ewm(span=ema_long, adjust=False).mean()

    # --- MACD and Signal Line ---
    df['MACD'] = df[f'EMA_{ema_short}'] - df[f'EMA_{ema_long}']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- Relative Strength Index (RSI) ---
    delta = df['Close'].diff()
    # Gains (only positive changes)
    gain = delta.where(delta > 0, 0)
    # Losses (absolute value of negative changes)
    loss = -delta.where(delta < 0, 0)
    # Use a rolling window of 14 periods (default) to compute average gain and loss
    window_length = 14
    avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()
    # Avoid division by zero
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- Average True Range (ATR) ---
    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df['Close'].shift())
    low_close_prev = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14, min_periods=14).mean()

    # --- Price Returns and Lag Features ---
    df['Return'] = df['Close'].pct_change()
    # Create lagged returns (e.g., 1-day and 2-day lag)
    df['Lag_1'] = df['Return'].shift(1)
    df['Lag_2'] = df['Return'].shift(2)
    
    # Optionally drop rows with NaN values generated from rolling calculations
    df = df.dropna().reset_index(drop=True)
    
    return df

def main():
    # Load your processed data
    df = pd.read_csv('data/processed/sp500_20years_processed.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Compute additional technical indicators
    df = compute_technical_indicators(df)
    
    # Save the enriched dataset
    df.to_csv('data/processed/sp500_20years_features.csv', index=False)
    print("Feature engineering complete. Data saved to 'data/processed/sp500_20years_features.csv'")

if __name__ == "__main__":
    main()
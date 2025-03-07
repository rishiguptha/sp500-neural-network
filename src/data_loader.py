import yfinance as yf
import pandas as pd

def download_data(ticker="^GSPC", period="20y", filename="data/raw/sp500_20years.csv"):
    """
    Downloads historical data for the specified ticker and saves it as CSV.
    """
    data = yf.download(ticker, period=period)
    data.to_csv(filename)
    print(f"Downloaded and saved S&P500 data to '{filename}'")
    return data

def load_data_from_csv(filepath="data/raw/sp500_20years.csv"):
    """
    Loads CSV data into a DataFrame.
    Assumes the CSV has a multi-index header and the first column as the Date index.
    """
    df = pd.read_csv(filepath, header=[0,1], index_col=0, parse_dates=True)
    return df

import pandas as pd
import numpy as np
import yfinance as yf

import pandas as pd
import numpy as np
import yfinance as yf
import os

def load_and_preprocess_data(raw_filepath="data/raw/sp500_20years.csv", 
                             processed_filepath="data/processed/sp500_20years_processed.csv"):
    """
    Loads S&P500 data from CSV (downloading it if necessary), fills missing values,
    computes log returns, and creates a binary target where a good day is defined as:
    tomorrow's closing price > today's closing price.
    
    The processed DataFrame is saved to the processed_filepath for future use.
    """
    # Ensure that the directories exist
    os.makedirs(os.path.dirname(raw_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    
    try:
        df = pd.read_csv(raw_filepath, header=[0, 1], index_col=0, parse_dates=True)
    except FileNotFoundError:
        data = yf.download("^GSPC", period="20y")
        data.to_csv(raw_filepath)
        df = pd.read_csv(raw_filepath, header=[0, 1], index_col=0, parse_dates=True)

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Fill missing values: forward then backward fill (avoid deprecated method)
    df = df.ffill().bfill()
    
    # Calculate log returns: ln(Close_t / Close_{t-1})
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()  # drop rows with NaN introduced by shift
    
    # Create target: 1 if next day's Close > today's Close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()  # drop the last row as target is undefined
    
    # Save processed DataFrame to processed folder for future use
    df.to_csv(processed_filepath)
    print(f"Processed data saved to '{processed_filepath}'")
    
    return df
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(df, exclude_columns=['Date', 'Target']):
    """
    Scales numeric features in the dataframe using StandardScaler.
    
    Parameters:
        df (pd.DataFrame): DataFrame with feature-engineered data.
        exclude_columns (list): Columns that should not be scaled.
    
    Returns:
        df_scaled (pd.DataFrame): DataFrame with scaled numeric features.
        scaler (StandardScaler): Fitted scaler (useful for later transforming new data).
    """
    # Identify columns to scale: all numeric columns except those we want to exclude
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_to_scale = [col for col in numeric_cols if col not in exclude_columns]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    return df_scaled, scaler

def main():
    # Load the feature engineered data
    df = pd.read_csv('data/processed/sp500_20years_features.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Scale the features (excluding Date and Target if needed)
    df_scaled, scaler = scale_features(df)
    
    # Save the scaled features
    df_scaled.to_csv('data/processed/sp500_20years_features_scaled.csv', index=False)
    print("Scaling complete. Data saved to 'data/processed/sp500_20years_features_scaled.csv'")

if __name__ == "__main__":
    main()
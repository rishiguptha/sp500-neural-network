from data_loader import download_data, load_data_from_csv
from feature_engineering import calculate_log_returns, clean_log_returns, compute_cumulants, calculate_var
from utils import plot_histogram_with_normal, plot_qq

def main():
    # Download S&P 500 data and save CSV (if not already downloaded)
    download_data()  # Saves file to data/raw/sp500_20years.csv
    
    # Load data from CSV
    df = load_data_from_csv("data/raw/sp500_20years.csv")
    
    # Print missing value statistics
    missing_values = df.isnull().sum(axis=1)
    print("\nMissing Values Per Row:")
    print(missing_values)
    
    rows_with_missing = df[df.isnull().any(axis=1)]
    print("\nRows with Missing Values:")
    print(rows_with_missing)
    
    # Calculate log returns and add to DataFrame
    df = calculate_log_returns(df)
    
    # Extract and clean the Log_Returns series
    log_returns = clean_log_returns(df['Log_Returns'])
    
    # Display summary statistics
    print(log_returns.describe())
    print("Number of unique log return values:", log_returns.nunique())
    
    # Compute cumulants for log returns
    mu, sigma, skew_val, kurt_val = compute_cumulants(log_returns)
    print("Cumulants for Log_Returns:")
    print("Mean:", mu)
    print("Variance:", sigma**2)
    print("Skewness:", skew_val)
    print("Excess Kurtosis:", kurt_val)
    
    # Plot histogram with normal PDF overlay
    plot_histogram_with_normal(log_returns, mu, sigma, skew_val, kurt_val)
    
    # Plot Q–Q plot
    plot_qq(log_returns)
    
    # Calculate and print the 95% VaR
    var_95 = calculate_var(log_returns, percentile=5)
    print("Historical 95% VaR:", var_95)

if __name__ == "__main__":
    main()
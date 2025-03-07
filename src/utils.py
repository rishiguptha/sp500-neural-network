import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, probplot

def plot_histogram_with_normal(log_returns, mu, sigma, skew_val, kurt_val):
    """
    Plots a histogram of log returns with an overlaid Normal PDF and annotations.
    """
    plt.figure(figsize=(10, 6))
    count, bins, patches = plt.hist(log_returns, bins=50, density=True, alpha=0.6,
                                    color='skyblue', edgecolor='black')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2, label='Normal PDF')
    plt.axvline(mu, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean = {mu:.4f}')
    plt.text(xmin + (xmax - xmin) * 0.05, max(p) * 0.9,
             f"Std Dev = {sigma:.4f}\nSkewness = {skew_val:.4f}\nExcess Kurtosis = {kurt_val:.4f}",
             bbox=dict(facecolor='white', alpha=0.8))
    plt.title("Histogram of Log Returns with Normal PDF Overlay")
    plt.xlabel("Log Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_qq(log_returns):
    """
    Plots a Q–Q plot of log returns against a normal distribution.
    """
    plt.figure(figsize=(10, 6))
    probplot(log_returns, dist="norm", plot=plt)
    plt.title("Q–Q Plot of Log Returns")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Ordered Log Returns")
    plt.show()
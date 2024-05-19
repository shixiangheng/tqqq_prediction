import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def fetch_data(symbol, start_date, end_date):
    """
    Fetch historical data for a given symbol.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close']

def plot_moving_averages(data, short_window=40, long_window=100):
    """
    Plot the data along with its moving averages of specified windows and
    draw vertical lines when the short moving average crosses the long moving average.
    """
    # Calculate moving averages
    short_mavg = data.rolling(window=short_window, min_periods=1).mean()
    long_mavg = data.rolling(window=long_window, min_periods=1).mean()
    
    # Identify crossover points
    crossover_points = ((short_mavg.shift(1) < long_mavg.shift(1)) & (short_mavg > long_mavg)) | ((short_mavg.shift(1) > long_mavg.shift(1)) & (short_mavg < long_mavg))
    crossover_dates = data.index[crossover_points]

    # Plot the original data, short moving average, and long moving average
    plt.figure(figsize=(14, 7))
    plt.plot(data, label='Stock Price (Close)')
    plt.plot(short_mavg, label=f'Short MA ({short_window})')
    plt.plot(long_mavg, label=f'Long MA ({long_window})', color='red')
    
    # Plot vertical lines at crossover points
    for date in crossover_dates:
        plt.axvline(x=date, color='green', linestyle='--')

    plt.title(f'Moving Averages and Close Price for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    symbol = 'TQQQ'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=11*365)
    
    # Format dates in YYYY-MM-DD format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    data =fetch_data(symbol, '2014-11-01', '2019-01-01')
    
    # Plot the moving averages
    plot_moving_averages(data)

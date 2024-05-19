# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:12:17 2024

@author: shixiangheng
"""

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
    draw vertical lines when the rate of change of the difference between
    short and long moving averages changes from decreasing to increasing (red)
    and from increasing to decreasing (blue).
    """
    # Calculate moving averages
    short_mavg = data.rolling(window=short_window, min_periods=1).mean()
    long_mavg = data.rolling(window=long_window, min_periods=1).mean()
    
    # Calculate the difference between short and long moving averages
    mavg_diff = short_mavg - long_mavg
    
    # Calculate the rate of change of the difference
    mavg_diff_rate_change = mavg_diff.diff()
    
    # Find where the rate of change of the difference changes from decreasing to increasing (mark red)
    red_points = ((mavg_diff_rate_change.shift(1) < 0) & (mavg_diff_rate_change > 0))
    red_dates = data.index[red_points]
    
    # Find where the rate of change of the difference changes from increasing to decreasing (mark blue)
    blue_points = ((mavg_diff_rate_change.shift(1) > 0) & (mavg_diff_rate_change < 0))
    blue_dates = data.index[blue_points]

    # Plot the original data, short moving average, and long moving average
    plt.figure(figsize=(14, 7))
    plt.plot(data, label='Stock Price (Close)')
    plt.plot(short_mavg, label=f'Short MA ({short_window})')
    plt.plot(long_mavg, label=f'Long MA ({long_window})', color='red')
    
    # Plot red vertical lines at points where the rate of change goes from decreasing to increasing
    for date in red_dates:
        plt.axvline(x=date, color='red', linestyle='--')
        
    # Plot blue vertical lines at points where the rate of change goes from increasing to decreasing
    for date in blue_dates:
        plt.axvline(x=date, color='blue', linestyle='--')

    plt.title(f'Moving Averages and Close Price for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    symbol = 'TQQQ'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Format dates in YYYY-MM-DD format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    data = fetch_data(symbol, '2021-11-01', '2023-01-01')
    
    short_window=40
    long_window=50
    # Plot the moving averages
    plot_moving_averages(data,short_window, long_window)

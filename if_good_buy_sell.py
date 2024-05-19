# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 00:02:21 2024

@author: shixiangheng
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_data(symbol, start_date, end_date):
    """
    Fetch historical data for a given symbol.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close']

def plot_recent_moving_averages(data, symbol, short_window=40, long_window=100, plot_months=6):
    """
    Plot the recent data (last plot_months months) along with its moving averages of specified windows
    and draw vertical lines when the rate of change of the difference between
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

    # Determine the plot start date (plot_months months before the last date in data)
    plot_start_date = data.index.max() - pd.DateOffset(months=plot_months)

    # Filter data for plotting
    plot_data = data[plot_start_date:]
    plot_short_mavg = short_mavg[plot_start_date:]
    plot_long_mavg = long_mavg[plot_start_date:]
    plot_red_dates = red_dates[red_dates >= plot_start_date]
    plot_blue_dates = blue_dates[blue_dates >= plot_start_date]

    # Plot the filtered data, short moving average, and long moving average
    plt.figure(figsize=(14, 7))
    plt.plot(plot_data, label='Stock Price (Close)')
    plt.plot(plot_short_mavg, label=f'Short MA ({short_window})')
    plt.plot(plot_long_mavg, label=f'Long MA ({long_window})', color='red')
    
    # Plot red vertical lines at points where the rate of change goes from decreasing to increasing
    for date in plot_red_dates:
        plt.axvline(x=date, color='red', linestyle='--')
        
    # Plot blue vertical lines at points where the rate of change goes from increasing to decreasing
    for date in plot_blue_dates:
        plt.axvline(x=date, color='blue', linestyle='--')

    plt.title(f'Recent 3-Month Moving Averages and Close Price for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    print('last Buy Date:', plot_red_dates[-1])
    print('last Buy price', data[plot_red_dates[-1]])
    print('last Sell Date:', plot_blue_dates[-1])
    print('last Sell price', data[plot_blue_dates[-1]])
    return data[plot_blue_dates[-1]]

if __name__ == "__main__":
    symbol = 'TQQQ'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1*365)  # Fetch 2 years of data for MA calculations
    
    # Fetch historical data
    data = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Ask user for the expected close price for today and append it to the data
    expected_close_price = float(input("Enter the expected close price for today: "))
    my_buy_price=float(input("Enter the current buy price for: "))
    
    today_str = end_date.strftime('%Y-%m-%d')  # Ensure it's in the correct format
    if today_str not in data.index:
        data.at[today_str] = expected_close_price  # Append the new data point
    #else:
    #    data[today_str] = expected_close_price  # Update the existing data point if today's data already exists
    
    # Ensure the data's index is a datetime index
    data.index = pd.to_datetime(data.index)
    
    # Plot the recent 3 months with the moving averages
    last_sell_price=plot_recent_moving_averages(data, symbol)
    print(data[-10:])
    print("************Suggestions:**************")
    if (data[today_str]/my_buy_price)>=1.5 or \
        (data[today_str]/my_buy_price)<=0.98 or \
            last_sell_price<my_buy_price:
        print('Sell it!')
    else:
        print("Hold")

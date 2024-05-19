# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:10:54 2024

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
    return data

def prepare_signals(data, short_window=40, long_window=100):
    """
    Prepare the buy and sell signals based on the change in direction of the difference between MA40 and MA100.
    """
    data['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    data['mavg_diff'] = data['short_mavg'] - data['long_mavg']
    data['diff_change'] = data['mavg_diff'].diff()
    #data['diff_change'] = data['diff_change'].diff()
    data['signal'] = 0.0
    # Buy signal: when diff changes from negative to positive
    # Sell signal: when diff changes from positive to negative
    data['signal'] = np.where(data['diff_change'] > 0, 1.0, np.where(data['diff_change'] < 0, -1.0, 0.0))
    data['positions'] = data['signal'].diff()
    return data

def execute_trades(data, initial_capital=10000.0):
    position = 0
    cash = initial_capital
    portfolio_values = []
    last_trade_date=None
    
    for date, row in data.iterrows():
        notbuy=0
        #if last_trade_date is not None and (date - last_trade_date).days < 30:
         #   notbuy=1
            #portfolio_value = cash + position * row['Close']
            #portfolio_values.append(portfolio_value)
            #continue
        
        
        if row['positions'] == 2.0 and notbuy==0:  # From no position to buy
        #if row['diff_change']>0:
            if cash > 0:
                position = cash / row['Close']
                cash = 0.0
                last_trade_date = date
        elif row['positions'] == -2.0 and notbuy==0:  # From buy to sell
            if position > 0:
                cash = position * row['Close']
                position = 0.0
                last_trade_date = date
        portfolio_value = cash + position * row['Close']
        portfolio_values.append(portfolio_value)

    data['portfolio_value'] = portfolio_values
    return data

def plot_performance(data):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('TQQQ Price', color=color)
    ax1.plot(data['Close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Strategy Portfolio Value', color=color)  # we already handled the x-label with ax1
    ax2.plot(data['portfolio_value'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    '''
    # Adding vertical lines for buy and sell signals
    for date, row in data.iterrows():
        if row['positions'] == 2.0:  # Buy signal
            ax1.axvline(x=date, color='blue', linestyle='--')
        elif row['positions'] == -2.0:  # Sell signal
            ax1.axvline(x=date, color='black', linestyle='--')
    '''
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('TQQQ Trading Strategy Performance')
    plt.show()

if __name__ == "__main__":
    symbol = 'TQQQ'
    end_date = datetime.now()
    #end_date=datetime(2024, 1, 1, 22, 35, 17, 747174)
    #fetch_data('TQQQ', '2021-01-01', '2022-01-01')
    start_date = end_date - timedelta(days=365*2.2)
    
    data = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    data_with_signals = prepare_signals(data)
    data_with_trades = execute_trades(data_with_signals)
    
    plot_performance(data_with_trades)

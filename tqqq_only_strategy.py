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
    Prepare the buy and sell signals based on the crossing of MA40 and MA100.
    """
    data['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Create a 'signal' for when to buy (1.0) and when to sell (-1.0)
    data['signal'] = 0.0
    data['signal'][short_window:] = np.where(data['short_mavg'][short_window:] > data['long_mavg'][short_window:], 1.0, 0.0)
    data['positions'] = data['signal'].diff()
    return data

def execute_trades(data, initial_capital=10000.0):
    """
    Execute trades based on the signals and calculate the portfolio value over time.
    """
    position = 0
    cash = initial_capital
    portfolio_values = []

    for date, row in data.iterrows():
        if (row['positions'])!=0:
            print(row['positions'])
        if row['positions'] == 1.0:  # Buy signal
            if cash > 0:
                position = cash / row['Close']
                cash = 0.0
        elif row['positions'] == -1.0:  # Sell signal
            if position > 0:
                cash = position * row['Close']
                position = 0.0
        portfolio_value = cash + position * row['Close']
        portfolio_values.append(portfolio_value)

    data['portfolio_value'] = portfolio_values
    return data

def plot_performance(data):
    """
    Plot the TQQQ price and the strategy's portfolio value over time, including buy/sell signals.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.plot( data['portfolio_value'], label='Strategy Portfolio Value', color='green')
    ax2.plot( data['Close'], label='TQQQ Price', color='gray', alpha=0.3)

    # Plot buy/sell signals
    for date, row in data.iterrows():
        if row['positions'] == 1.0:  # Buy signal
            ax1.axvline(x=date, color='blue', linestyle='--', lw=1)
        elif row['positions'] == -1.0:  # Sell signal
            ax1.axvline(x=date, color='black', linestyle='--', lw=1)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value', color='green')
    ax2.set_ylabel('TQQQ Price', color='gray')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('TQQQ Price and Strategy Performance')
    plt.show()

if __name__ == "__main__":
    symbol = 'TQQQ'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    data = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    data_with_signals = prepare_signals(data)
    data_with_trades = execute_trades(data_with_signals)
    
    plot_performance(data_with_trades)



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
    buy_price = 0  # To track the price at which we last bought the stock
    
    for date, row in data.iterrows():
        notbuy = 0
        current_price = row['Close']
        # Adjust strategy based on MA conditions
        if row['short_mavg'] < row['long_mavg']:
            # Adjust for specific strategy when MA40 is below MA100
            short_window, long_window = 20, 80
            profit_to_buy, profit_to_sell = -2, 5
        else:
            # Default strategy settings
            short_window, long_window = 40, 100
            profit_to_buy, profit_to_sell = -2, 50

        # Recalculate signals based on the adjusted strategy
        data = prepare_signals(data, short_window, long_window)

        # Execute trades based on the current position, cash, and profit/loss calculations
        if position > 0:
            current_price = row['Close']
            profit_loss_percent = ((current_price - buy_price) / buy_price) * 100

            if profit_loss_percent <= profit_to_buy or profit_loss_percent >= profit_to_sell:
                cash += position * current_price
                position = 0
                buy_price = 0  # Resetting buy_price since we sold
                notbuy = 1

        if not notbuy:
            # From no position to buy
            if row['positions'] == 2.0:  
                if cash > 0:
                    position = cash / current_price
                    cash = 0.0
                    buy_price = current_price
            # From buy to sell
            elif row['positions'] == -2.0:  
                if position > 0:
                    cash += position * current_price
                    position = 0
                    buy_price = 0

        portfolio_value = cash + position * current_price
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
    end_date=datetime(2024, 1, 1, 22, 35, 17, 747174)
    #fetch_data('TQQQ', '2021-01-01', '2022-01-01')
    start_date = end_date - timedelta(days=365*2.5)
    
    data = fetch_data(symbol, '2022-01-01', '2023-01-01')
    data_with_signals = prepare_signals(data)
    data_with_trades = execute_trades(data_with_signals)
    
    plot_performance(data_with_trades)

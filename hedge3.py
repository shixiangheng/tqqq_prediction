# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:43:55 2024

@author: shixiangheng
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fetch_data(symbol, start_date, end_date):
    """
    Fetch historical data for a given symbol.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close']

def simple_moving_average_strategy(data, short_window, long_window):
    """
    Generate trading signals based on moving averages.
    """
    signals = pd.DataFrame(index=data.index)
    signals['data'] = data
    signals['short_mavg'] = data.rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data.rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals
def execute_hedged_trades(data_tqqq, data_gld, initial_capital=10000.0):
    cash = initial_capital
    position_tqqq = position_gld = 0.0
    equity_curve = []
    trades_info = []  # List to hold trade information

    for date, row in data_tqqq.iterrows():
        gld_row = data_gld.loc[date]
        # Buy GLD
        if row['positions'] == 1.0 and cash > 0:
            position_gld = cash / gld_row['data']
            cash = 0.0
            trades_info.append([date, 'BUY', 'GLD', position_gld, gld_row['data'], 0, cash])
        # Sell GLD and buy TQQQ
        elif row['positions'] == -1.0 and position_gld > 0:
            cash = position_gld * gld_row['data']
            position_tqqq = cash / row['data']
            trades_info.append([date, 'SELL', 'GLD', position_gld, gld_row['data'], cash, 0])
            cash = 0.0
            position_gld = 0.0
            trades_info.append([date, 'BUY', 'TQQQ', position_tqqq, row['data'], 0, cash])
        
        # Update equity curve
        total_value = cash + position_tqqq * row['data'] + position_gld * gld_row['data']
        equity_curve.append(total_value)

    # Final update to sell TQQQ if still holding
    if position_tqqq > 0:
        final_tqqq_price = data_tqqq.iloc[-1]['data']
        cash = position_tqqq * final_tqqq_price
        trades_info.append([data_tqqq.index[-1], 'SELL', 'TQQQ', position_tqqq, final_tqqq_price, cash, 0])
        equity_curve[-1] = cash  # Update the last value in equity curve

    profit = cash - initial_capital
    profit_rate = (profit / initial_capital) * 100
    
    # Adjust columns in DataFrame construction to include stock price at transaction
    trades_df = pd.DataFrame(trades_info, columns=['Date', 'Action', 'Stock', 'Shares', 'Stock Price at Transaction', 'Cash After Trade', 'Position After Trade'])
    
    return trades_df, equity_curve, profit, profit_rate



# Fetch data
data_tqqq = fetch_data('TQQQ', '2019-01-01', '2024-01-01')
data_gld = fetch_data('GLD', '2019-01-01', '2024-01-01')

# Generate signals
signals_tqqq = simple_moving_average_strategy(data_tqqq, 40, 100)
signals_gld = simple_moving_average_strategy(data_gld, 40, 100)

# Execute trades with hedging
#trade_dates, trade_actions, equity_curve, profit, profit_rate = execute_hedged_trades(signals_tqqq, signals_gld)


# After executing trades with hedging
trades_df, equity_curve, profit, profit_rate = execute_hedged_trades(signals_tqqq, signals_gld)

# Save trades information to an Excel file
trades_df.to_excel("trade_details.xlsx", index=False)
# Plot results
# Plot results
plt.figure(figsize=(14, 7))
plt.plot(equity_curve, label='Portfolio Value')
plt.title("Equity Curve with Trade Signals")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()




print(f"Final Portfolio Value: ${equity_curve[-1]:.2f}")
print(f"Total Profit: ${profit:.2f}")
print(f"Profit Rate: {profit_rate:.2f}%")

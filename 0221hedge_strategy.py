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

def execute_trades(data, initial_capital=10000.0):
    """
    Simulate trades and calculate the portfolio's performance.
    """
    position = 0.0
    cash = initial_capital
    equity_curve = []

    for _, row in data.iterrows():
        # Buy signal
        if row['positions'] == 1.0:
            position = cash / row['data']
            cash = 0.0
        
        # Sell signal
        elif row['positions'] == -1.0 and position > 0:
            cash = position * row['data']
            position = 0.0
        
        total_value = cash + position * row['data']
        equity_curve.append(total_value)

    data['equity_curve'] = equity_curve
    profit = equity_curve[-1] - initial_capital
    profit_rate = (profit / initial_capital) * 100

    return profit, profit_rate, data

# Example usage
symbols = {'TQQQ': 10000, 'GLD': 10000}
results = {}

for symbol, initial_capital in symbols.items():
    data = fetch_data(symbol, '2019-01-01', '2024-01-01')
    signals = simple_moving_average_strategy(data, 40, 100)
    profit, profit_rate, signals_with_equity = execute_trades(signals, initial_capital)
    results[symbol] = {'Profit': profit, 'Profit Rate': profit_rate}

    # Plotting
    plt.figure(figsize=(12, 6))
    signals_with_equity['equity_curve'].plot()
    plt.title(f"{symbol} Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.show()

    print(f"{symbol} Profit: ${profit:.2f}, Profit Rate: {profit_rate:.2f}%")

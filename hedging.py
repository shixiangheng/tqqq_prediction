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
    """
    Execute trades with hedging between TQQQ and GLD.
    """
    cash = initial_capital
    position = 0.0
    equity_curve = []
    trade_dates = []
    trade_actions = []

    for date, row in data_tqqq.iterrows():
        # Determine which asset to interact with
        if row['positions'] == 1.0:  # Buy TQQQ
            if cash > 0:  # Only buy if we have cash
                position = cash / row['data']
                cash = 0.0
                trade_dates.append(date)
                trade_actions.append('BUY TQQQ')
        elif row['positions'] == -1.0:  # Sell TQQQ
            if position > 0:  # Only sell if we hold TQQQ
                cash = position * row['data']
                position = 0.0
                trade_dates.append(date)
                trade_actions.append('SELL TQQQ')
                # After selling TQQQ, attempt to buy GLD with all available cash
                gld_row = data_gld.loc[date]
                position = cash / gld_row['data']
                cash = 0.0
                trade_dates.append(date)
                trade_actions.append('BUY GLD')
                
        # Update equity curve for TQQQ
        total_value = cash + position * row['data']
        equity_curve.append(total_value)

    # Final update using GLD data to sell GLD if still holding
    if position > 0:  # If holding GLD at the end
        final_gld_price = data_gld.iloc[-1]['data']
        cash = position * final_gld_price
        position = 0.0
        trade_dates.append(data_gld.index[-1])
        trade_actions.append('SELL GLD')
        equity_curve[-1] = cash  # Update the last value in equity curve

    profit = cash - initial_capital
    profit_rate = (profit / initial_capital) * 100
    
    return trade_dates, trade_actions, equity_curve, profit, profit_rate

# Fetch data
data_tqqq = fetch_data('GLD', '2021-01-01', '2022-01-01')
data_gld = fetch_data('TQQQ', '2021-01-01', '2022-01-01')

# Generate signals
signals_tqqq = simple_moving_average_strategy(data_tqqq, 40, 100)
signals_gld = simple_moving_average_strategy(data_gld, 40, 100)

# Execute trades with hedging
trade_dates, trade_actions, equity_curve, profit, profit_rate = execute_hedged_trades(signals_tqqq, signals_gld)

# Plot results
# Plot results
plt.figure(figsize=(14, 7))
plt.plot(equity_curve, label='Portfolio Value')
for date, action in zip(trade_dates, trade_actions):
    color = 'green' if 'BUY' in action else 'red'
    plt.plot(trade_dates.index(date), equity_curve[trade_dates.index(date)], marker='o', color=color, label=action if trade_dates.index(date) == 0 else "")
plt.title("Equity Curve with Trade Signals")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()




print(f"Final Portfolio Value: ${equity_curve[-1]:.2f}")
print(f"Total Profit: ${profit:.2f}")
print(f"Profit Rate: {profit_rate:.2f}%")

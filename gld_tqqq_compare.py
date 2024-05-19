import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Define the tickers
tickers = ['GLD', 'TQQQ']

# Define the period
end_date = pd.Timestamp.today()  # Today's date
start_date = end_date - pd.DateOffset(years=5)  # 5 years before today

# Initialize an empty DataFrame to hold the close prices
close_prices = pd.DataFrame()

# Fetch the historical data
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    close_prices[ticker] = data['Close']

# Plotting
plt.figure(figsize=(14, 7))

for ticker in tickers:
    plt.plot(close_prices[ticker], label=ticker)

# Adding plot title and labels
plt.title('GLD vs TQQQ Close Price Over the Last 5 Years')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Show plot
plt.show()

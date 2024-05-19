# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:16:22 2024

@author: shixiangheng
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:31:40 2024

@author: shixiangheng
"""




import tkinter as tk
from tkinter import ttk
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def fetch_data(symbol, start_date, end_date):
    """
    Fetch historical data for a given symbol.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close']


def plot_recent_moving_averages(master, info_frame, data, symbol, short_window=40, long_window=100, plot_months=6):
    # Clear the master frame
    for widget in master.winfo_children():
        widget.destroy()
    
    for widget in info_frame.winfo_children():
        widget.destroy()
    # Calculate moving averages and other necessary data points
    short_mavg = data.rolling(window=short_window, min_periods=1).mean()
    long_mavg = data.rolling(window=long_window, min_periods=1).mean()
    mavg_diff = short_mavg - long_mavg
    mavg_diff_rate_change = mavg_diff.diff()
    red_points = ((mavg_diff_rate_change.shift(1) < 0) & (mavg_diff_rate_change > 0))
    red_dates = data.index[red_points]
    blue_points = ((mavg_diff_rate_change.shift(1) > 0) & (mavg_diff_rate_change < 0))
    blue_dates = data.index[blue_points]
    plot_start_date = data.index.max() - pd.DateOffset(months=plot_months)

    # Prepare plot data
    plot_data = data[plot_start_date:]
    plot_short_mavg = short_mavg[plot_start_date:]
    plot_long_mavg = long_mavg[plot_start_date:]
    plot_red_dates = red_dates[red_dates >= plot_start_date]
    plot_blue_dates = blue_dates[blue_dates >= plot_start_date]
    
    
    #last_buy_date=red_dates[-1]
    #last_sell_date=blue_dates[-1]
    
    # Plotting
    fig = Figure(figsize=(10, 4), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.plot(plot_data, label='Stock Price (Close)')
    plot1.plot(plot_short_mavg, label=f'Short MA ({short_window})')
    plot1.plot( plot_long_mavg, label=f'Long MA ({long_window})', color='red')

    for date in plot_red_dates:
        plot1.axvline(x=date, color='red', linestyle='--')
    for date in plot_blue_dates:
        plot1.axvline(x=date, color='blue', linestyle='--')

    plot1.legend()
    plot1.set_title(f'Recent Moving Averages and Close Price for {symbol}')
    plot1.set_xlabel('Date')
    plot1.set_ylabel('Price')

    # Create a canvas and put the plot on it
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    df=pd.DataFrame(data, columns=['Close'])
    df.index = pd.to_datetime(data.index)
    # Display last 10 days' stock prices
    recent_prices = df['Close'].tail(10)
    recent_prices_str = "\n".join([f"{date.strftime('%Y-%m-%d')}: {price:.2f}" for date, price in recent_prices.items()])
    recent_prices_label = tk.Label(info_frame, text=f"Recent 10 days' prices:\n{recent_prices_str}", justify=tk.LEFT)
    recent_prices_label.pack()
    
    # Assuming the last relevant action based on the last date in red/blue dates for suggestion logic
    '''
    try:
        last_sell_price = None
        if len(plot_blue_dates) > 0:
            last_sell_date = plot_blue_dates[-1]
            last_sell_price = data.loc[last_sell_date]
        last_buy_signal = (len(plot_red_dates) > 0 and (len(plot_blue_dates) == 0 or plot_red_dates[-1] > plot_blue_dates[-1]))
        return last_sell_price, last_buy_signal  # True if buy signal is more recent, False otherwise
    '''
    try:
        #last_sell_price = None
        if len(plot_blue_dates) > 0:
            last_sell_date = blue_dates[-1]
            last_sell_price = data.loc[last_sell_date]
            last_buy_date= red_dates[-1]
            last_buy_price = data.loc[last_buy_date]
        return last_sell_date,last_sell_price,last_buy_date,last_buy_price
    except IndexError:  # Handle cases where there are no buy/sell signals
        return None, None, None, None

def on_submit():
    #symbol = symbol_entry.get()
    symbol='TQQQ'
    
    expected_close_price = float(expected_price_entry.get())
    if buy_price_entry.get()!='':
        my_buy_price = float(buy_price_entry.get())
    else:
        my_buy_price=None
    #my_buy_price = float(buy_price_entry.get()) if buy_price_entry !='' else None 
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    today_str = end_date.strftime('%Y-%m-%d')
    if today_str not in data.index:
        data.at[today_str] = expected_close_price
    data.index = pd.to_datetime(data.index)
    
    # Plot the recent 3 months with the moving averages
    last_sell_date,last_sell_price,last_buy_date,last_buy_price=plot_recent_moving_averages(canvas,info_frame, data, symbol)
    
    # Your suggestion logic (simplified for brevity)
    suggestion_text.set("Evaluating suggestion...")
    #last_sell_price = data[-1]  # Simplified logic; adapt according to your needs
    if last_sell_price is None:
        last_sell_price=9999
    
    if my_buy_price:
        upbound=round(my_buy_price*1.5,2)
        lowbound=round(my_buy_price*0.98,2)
        bound=(lowbound,upbound)
        if last_sell_date<last_buy_date:  # case 1: already bought. after buy date
            # profit rate analysis
            if (expected_close_price/my_buy_price) >= 1.5 or (expected_close_price/my_buy_price) <= 0.98:
                suggestion_text.set('Sell it! It reached (-2,50) profit point. The bound is '+str(bound)+ '. The last sell price is '+str(last_sell_price))
            else:
                suggestion_text.set("Hold it! The last buy price is "+str(last_buy_price))
        else:    # last action date is sell date(but still hold now)
            # profit rate analysis
            if (expected_close_price/my_buy_price) >= 1.5 or (expected_close_price/my_buy_price) <= 0.98:
                suggestion_text.set('Sell it! It reached (-2,50) profit point. The bound is '+str(bound)+ '. The last sell price is '+str(last_sell_price))
            elif str(last_sell_date)[:10]==str(today_str)[:10]:  # today is the sell date
                suggestion_text.set('Sell it! Today is the sell date! The last sell price is '+str(last_sell_price))
            elif expected_close_price<last_sell_price*(1.00):
                suggestion_text.set('Sell it! It is below last sell price. The last sell price is '+str(last_sell_price))
            else:
                suggestion_text.set("Hold it! The last sell price is "+str(last_sell_price))
    else: # not buy
        if last_sell_date<last_buy_date or last_buy_date==today_str :# last action date is buy date or today is buy date
            suggestion_text.set("Buy it now! The last buy price is "+str(last_buy_price))
        else: # last action date is sell date
            suggestion_text.set("Wait for next buy date")
            

root = tk.Tk()
root.title("Stock Analysis Tool")


# Create a main frame that will contain the plot and info frames
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Define the plot frame (left)
plot_frame = ttk.Frame(main_frame)
plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Define the info frame (right)
info_frame = ttk.Frame(main_frame)
info_frame.pack(side=tk.RIGHT, fill=tk.Y)
'''
symbol_label = ttk.Label(frame, text="Symbol:")
symbol_label.pack()
symbol_entry = ttk.Entry(frame)
symbol_entry.pack()
'''
expected_price_label = ttk.Label(main_frame, text="Expected Close Price:")
expected_price_label.pack()
expected_price_entry = ttk.Entry(main_frame)
expected_price_entry.pack()

buy_price_label = ttk.Label(main_frame, text="Buy Price:")
buy_price_label.pack()
buy_price_entry = ttk.Entry(main_frame)
buy_price_entry.pack()

submit_button = ttk.Button(main_frame, text="Submit", command=on_submit)
submit_button.pack()

# Canvas for the plot
# Canvas for the plot (modify to use plot_frame instead of root)
fig_canvas = FigureCanvasTkAgg(Figure(figsize=(5, 4)), master=plot_frame)  # A tk.DrawingArea.
canvas = fig_canvas.get_tk_widget()
canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Label for showing suggestions
suggestion_text = tk.StringVar()
suggestion_label = ttk.Label(root, textvariable=suggestion_text)
suggestion_label.pack(side=tk.BOTTOM)

root.mainloop()
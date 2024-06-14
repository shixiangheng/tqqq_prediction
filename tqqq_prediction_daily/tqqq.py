import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics
import math

def max_index(lst):
    """Returns the index of the maximum element in the given list"""
    max_val = max(lst)
    max_idx = lst.index(max_val)
    return max_idx

def choose_v(keys,values,model_l,my_dict,X_input,lr_model,lag):
    date_string=''
    reverse=0
    '''
    if max(values)<0.5:
        reverse=1
        for j in range(len(values)):
            if values[j]<0.5:
                values[j]=1-values[j]
    '''
    reverse_index=[]
    for j in range(len(values)):
        if values[j]<0.5:
            values[j]=1-values[j]
            reverse_index.append(j)
    i=max_index(values)
    model=model_l[i]
    X_today=X_input.values
    time=X_input.index[0]
    date_string = time.strftime('%Y-%m-%d %H:%M:%S')
    date_string =date_string[:10]
    if model==lr_model:
        
        #print(df['Close'][-1])
        #lr_model.fit()
        lr_pred = np.where(lr_model.predict(X_today) > 0.5, 1, 0)
        
       
        result= lr_pred[0]
    else:
        #model.fit(X_train, y_train)
        model_pred = model.predict(X_today)
        result=model_pred[0]
    #print(keys[i])
    if i in reverse_index:
        if result==1:
            result=0
        else:  # when result==0
            result=1
    return (result,values[i])
def find_index_closest_to_0_5(lst):
    """从列表中选出离0.5最远的数字的index。

    Args:
        lst: 包含数字的列表。

    Returns:
        离0.5最远的数字的index。
    """
    # 将列表转换为numpy数组
    arr = np.array(lst)

    # 找到与0.5的距离最远的数字的索引
    index = np.abs(arr - 0.5).argmax()

    return index
def main_draw(lag=1):
    last=-1
    #year='5y'
    year='3y'
    # 从yfinance API获取股票、国债和黄金数据
    tqqq = yf.Ticker('TQQQ')
    tqqq_df = tqqq.history(period=year)
    bond = yf.Ticker('^TNX')
    bond_df = bond.history(period=year)
    gold = yf.Ticker('GC=F')
    gold_df = gold.history(period=year)
    tesla = yf.Ticker('TSLA')
    tesla_df = tesla.history(period=year)
    
    # 计算五日均线
    tqqq_df['5_day_avg'] = tqqq_df['Close'].rolling(window=5).mean()
    bond_df['5_day_avg'] = bond_df['Close'].rolling(window=5).mean()
    gold_df['5_day_avg'] = gold_df['Close'].rolling(window=5).mean()
    tesla_df['5_day_avg'] = tesla_df['Close'].rolling(window=5).mean()
    
    # 合并所有数据
    df = pd.DataFrame(index=tqqq_df.index)
    df['Close']=tqqq_df['Close']
    
    df['Open'] = tqqq_df['Open']
    df['High'] = tqqq_df['High']
    df['Low'] = tqqq_df['Low']
    df['Volume'] = tqqq_df['Volume']
    df['TQQQ_5_day_avg'] = tqqq_df['5_day_avg']
    df['Bond_5_day_avg'] = bond_df['5_day_avg']
    df['Gold_5_day_avg'] = gold_df['5_day_avg']
    df['Tesla_5_day_avg'] = tesla_df['5_day_avg']
    
    # 去掉有缺失值的行
    df.dropna(inplace=True)
    o_df=df.copy()
    #df=df.iloc[:-1]  # remove today.  should remove this line (if at night), just now because it is in the trading time
    # used N-1 day for the end of sample data, so [:last]
    df=df.iloc[:last] 
    # 计算涨跌情况
    
    df['Price_Change'] = np.where(df['Close'].diff() > 0, 1, 0)
    
    # 用昨天的数据预测今天的收盘价涨跌情况
    df_close=df[['Close','Open', 'High', 'Low', 'Volume', 'TQQQ_5_day_avg', 'Bond_5_day_avg', 'Gold_5_day_avg', 'Tesla_5_day_avg']].iloc[:-1]
    
    o_X_df=o_df[['Open', 'Volume', 'TQQQ_5_day_avg', 'Bond_5_day_avg', 'Gold_5_day_avg', 'Tesla_5_day_avg']]
    
    
    # train data
    X_df=df[['Open', 'Volume', 'TQQQ_5_day_avg', 'Bond_5_day_avg', 'Gold_5_day_avg', 'Tesla_5_day_avg']].iloc[:-lag]
    X = df[['Open', 'Volume', 'TQQQ_5_day_avg', 'Bond_5_day_avg', 'Gold_5_day_avg', 'Tesla_5_day_avg']].iloc[:-lag].values
    y = df['Price_Change'].iloc[lag:].values
    
    today_Xdf=o_df[['Open', 'Volume', 'TQQQ_5_day_avg', 'Bond_5_day_avg', 'Gold_5_day_avg', 'Tesla_5_day_avg']].iloc[-1:,:]
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=88)
    
    # 线性回归模型
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = np.where(lr_model.predict(X_test) > 0.5, 1, 0)
    
    # 逻辑回归模型
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    logreg_pred = logreg_model.predict(X_test)
    
    # 随机森林模型
    # 5.8
    rf_l=[]
    rf_acu=[]
    for i in range(10):
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_l.append(rf_model)
        rf_acu.append(accuracy_score(y_test, rf_pred))
    rf_i=find_index_closest_to_0_5(rf_acu)
    rf_model=rf_l[rf_i]
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    '''
    # 时间序列模型 (ARIMA)
    ts_model = ARIMA(y_train, order=(1, 0, 0))
    ts_model_fit = ts_model.fit()
    ts_pred = np.where(ts_model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1) > 0, 1, 0)
    '''
    # 计算准确率
    lr_accuracy = accuracy_score(y_test, lr_pred)
    logreg_accuracy = accuracy_score(y_test, logreg_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    #ts_accuracy = accuracy_score(y_test, ts_pred)
    
    
    #print("Linear Regression Accuracy: {:.2f}%".format(lr_accuracy * 100))
    #print("Random Forest Accuracy: {:.2f}%".format(rf_accuracy * 100))
    #print("Time Series (ARIMA) Accuracy: {:.3f}%".format(ts_accuracy * 100))
    #print("Logistic Regression Accuracy: {:.3f}%".format(logreg_accuracy * 100))
    #print('--------')
    #t_df=X_df.iloc[-1,:]
    time=X_df.index[-1]
    
    date_str = time.strftime('%Y-%m-%d %H:%M:%S')
    date_str =date_str[:10]
    #print('Train Data Last Date: ', date_str)
    #print('--------')
    keys=['lr_accuracy', 'rf_accuracy' ,'logreg_accuracy']
    values=[lr_accuracy, rf_accuracy, logreg_accuracy]
    model_l=[lr_model,rf_model,logreg_model]
    name_model_l=['lr_model','rf_model','logreg_model']
    my_dict = dict(zip(keys, model_l))
    #X_input=X_df.iloc[-1:, :]
    #choose(keys,values,model_l,my_dict,today_Xdf,lr_model,lag)
    
    #print(today_Xdf)
    return choose_v(keys,values,model_l,my_dict,today_Xdf,lr_model,lag)

import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox


class GUI:

    def __init__(self):
        self.press=None
        self.root = tk.Tk()
        self.root.title("TQQQ Prediction")
        self.root.geometry("800x600")
        self.mean = None
        self.fig=None
        self.label = tk.Label(self.root, text="Enter number of days you would like to predict:")
        self.label.pack()
        self.press = None
        self.x0 = 0
        self.y0 = 0
        self.entry = tk.Entry(self.root)
        self.entry.pack()

        self.button = tk.Button(self.root, text="Run", command=self.run)
        self.button.pack(pady=10)

        self.canvas = None
        self.root.mainloop()
    '''
    def on_press(self, event):
        # capture the current position of the mouse
        self.press = event.x, event.y
        self.x0 = event.x
        self.y0 =event.y
        #dx = event.x - self.press[0]
        #dy = event.y - self.press[1]
       # self.x0=self.x0 +dx
        #self.y0=self.y0 +dy
    '''
    
    def run(self):
        
        x,y,acu=self.plot()
        acu=[1.0]+acu
        emp_str=''
        for i in range(len(x)):
            acu[i]=round(acu[i]*100,3)
            emp_str=emp_str+'Price: '+str(y[i])+ ' \nAccuracy for the up-down: '+ str(acu[i]) + '% after ' + str(x[i]) + ' Day(s) \n'
        messagebox.showinfo("info", emp_str)
    
    def plot(self):
        fig = Figure(figsize=(50, 40), dpi=100)
        ax = fig.add_subplot(111)
        # remove any existing plots from the canvas
        if self.canvas is not None:
            self.canvas.get_tk_widget().pack_forget()
        
        n = int(self.entry.get()) #should be 20
        # change x y
        year='3d'
        # 从yfinance API获取股票、国债和黄金数据
        tqqq = yf.Ticker('TQQQ')
        tqqq_df = tqqq.history(period=year)
        price_data=tqqq_df['Close']
        numbers = [i for i in range(1, n+1)]
        xl=[i for i in range(0, n+1)]
#price=today_Xdf['Open'][0]
#price=27.04
        price=price_data[-1]
        price0=27.45
        price_l=[price]

        time=price_data.index[-1]
        date_string = time.strftime('%Y-%m-%d %H:%M:%S')
        date_string =date_string[:10]
        acu_list=[]

        for i in numbers:
            random_number = np.random.normal(loc=0, scale=math.sqrt(statistics.variance(price_data)))
            random_number=abs(random_number)
            up_down,acu=main_draw(i)
            acu=round(acu,4)
            acu_list.append(round(acu,4))
            print(price)
            print("Accuracy for the up-down: "+ str(acu) + ' after ' + str(i) + ' Day(s)')
            #print('here')
            if up_down>0:
                price= price+random_number
            else:
                price=price-random_number
    
            price_l.append(price)
        ax.set_xticks(range(0, n+1))   #ax.set_xticks(range(0, 21))
        x=list(range(0,n+1))   #x=list(range(0,21))
        y=price_l
        #x = np.linspace(-5, 5, 100)
        #y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - self.mean) ** 2))
        
        #x=[1,2,3,4]
        #y=[10,20,30,40]
        ax.set_xlabel('Days after '+date_string)
        ax.set_ylabel('Price')
        ax.set_title(f'Prediction')
        ax.plot(x, y)
    # Add scrolling and dragging functionality
        def on_scroll(event):
            #self.on_press(event)
            # get the current size of the figure in inches
            cur_size = fig.get_size_inches()
        # adjust the size based on the scroll direction (up or down)
            if event.button == 'up':
                cur_size *= 1.1
            else:
                cur_size *= 0.9
        # set the new size of the figure
            #self.on_press(event)
            fig.set_size_inches(cur_size)
        # redraw the canvas with the new size
            if self.canvas is not None:
                self.canvas.get_tk_widget().forget()

                self.canvas = FigureCanvasTkAgg(fig, master=self.root)
                
                self.canvas.get_tk_widget().place(x=self.x0, y=self.y0)

                self.canvas.draw()
                
        
        
        '''
        def on_motion(event):
    # only do something if a mouse button has been pressed
            if self.press is None:
                pass

    # calculate the change in x-coordinate from the press position to the current position
            dx = event.x - self.press[0]

    # get the current x-limits of the plot and adjust them by dx
            xlim = ax.get_xlim()
            new_xlim = (xlim[0] - dx, xlim[1] - dx)
            ax.set_xlim(new_xlim)

    # redraw the canvas with the updated plot
            self.canvas.draw()
        '''
        
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        #fig.canvas.mpl_connect('button_press_event', on_press)
        '''
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        '''
        if self.canvas is not None:
            self.canvas.get_tk_widget().pack_forget()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        fig.savefig('my_plot.png')
        return x,y,acu_list


if __name__ == "__main__":
    GUI()
    #gui.root.mainloop()

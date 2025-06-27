import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



    # 1. 从雅虎财经下载历史数据
    # 2. 只保留 'Close' 和 'Volume'
    # 3. 转为美东时区
    # 4. 对收盘价和成交量各自做 MinMax 归一化（[0,1]）
    # 返回：归一化后的 DataFrame + 两个 scaler（后面用来反归一化）
    
def fetch_and_scale(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period)
    df = df[['Close', 'Volume']].dropna()
    df.index = df.index.tz_convert('US/Eastern')
    scaler_c = MinMaxScaler(); scaler_v = MinMaxScaler()
    df['Close']  = scaler_c.fit_transform(df[['Close']])
    df['Volume'] = scaler_v.fit_transform(df[['Volume']])
    return df, scaler_c, scaler_v


 
    # 把连续的时序数据切成小“滑动窗口”：
    #   - X: [样本数, seq_length, 特征数]
    #   - y: [样本数,]，预测窗口之后的那个时刻的收盘价

def make_sequences(df, seq_length):
    data = df.values
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, 0])  # index 0 为 Close
    return np.array(X), np.array(y)
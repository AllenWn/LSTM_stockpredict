import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler



    # 1. 从雅虎财经下载历史数据
    # 2. 只保留 'Close' 和 'Volume'
    # 3. 转为美东时区
    # 4. 对收盘价和成交量各自做 MinMax 归一化（[0,1]）
    # 返回：归一化后的 DataFrame + 两个 scaler（后面用来反归一化）

def fetch_and_scale(ticker, interval, period, features=None):
    # df = yf.download(ticker, interval=interval, period=period)
    # df = df[['Close', 'Volume']].dropna()
    # df.index = df.index.tz_convert('US/Eastern')
    # scaler_c = MinMaxScaler(); scaler_v = MinMaxScaler()
    # df['Close']  = scaler_c.fit_transform(df[['Close']])
    # df['Volume'] = scaler_v.fit_transform(df[['Volume']])
    # return df, scaler_c, scaler_v

    df = yf.download(ticker, interval=interval, period=period)
    df.index = df.index.tz_convert('US/Eastern')
    if features is None:
        features = ['Close', 'Volume']
    df = df[features].dropna()

    scalers = {}
    for col in features:
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df, scalers



"""
    Load data from a local CSV file, select Close and Volume, convert timezone,
    and apply MinMax scaling.
    Returns:
        df_scaled: pd.DataFrame with scaled 'Close' and 'Volume'
        scaler_close: fitted MinMaxScaler for 'Close'
        scaler_volume: fitted MinMaxScaler for 'Volume'
"""

def load_and_scale(file_path : str, features=None):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    if features is None:
        features = ['Close', 'Volume']
    df = df[features].dropna()

    scalers = {}
    for col in features:
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df, scalers

 
    # 把连续的时序数据切成小“滑动窗口”：
    #   - X: [样本数, seq_length, 特征数]
    #   - y: [样本数,]，预测窗口之后的那个时刻的收盘价

"""
    data: numpy array shape [T, num_features]
    seq_length: 窗口长度
    target_col: y 对应 data 的哪一列索引，这里 0 表示 'Close'
"""

def make_sequences(data: np.ndarray, seq_length: int, target_col: int = 0):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])           # shape = (seq_length, num_features)
        y.append(data[i+seq_length, target_col])    # 只取下一时刻的 Close
    return np.array(X), np.array(y)

# def make_sequences(df, seq_length):
#     data = df.values
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i+seq_length, :])
#         y.append(data[i+seq_length, 0])  # index 0 为 Close
#     return np.array(X), np.array(y)
import yaml
import joblib
import logging
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from datetime import datetime
from data.prepare_data import make_sequences
from models.lstm_model import LSTMModel
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import train
import predict


cfg = yaml.safe_load(open('config.yaml'))

test_csv     = cfg['test_data_path']
features     = cfg['features']
target_col   = cfg['target_col']
seq_len      = cfg['seq_length']
future_hours = cfg['future_hours']
target_name  = cfg['features'][target_col]

# 2) 准备日志文件
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# log_path = log_dir / f"run_{timestamp}.txt"
log_path = log_dir / f"run_001.txt"

# 3) 将重要参数写入日志
with open(log_path, 'w') as log_file:
    log_file.write(f"Experiment Start: {timestamp}\n")
    log_file.write("Configuration Parameters:\n")
    for key in ['features', 'target_col', 'seq_length',
                'hidden_size', 'num_layers', 'lr',
                'epochs', 'test_size', 'n_recent']:
        value = cfg.get(key, 'N/A')
        log_file.write(f"{key}: {value}\n")


with open(log_path, 'a') as log_file:
    log_file.write("\n=== Begin Training ===\n")
    with redirect_stdout(log_file): 
        train.main()



# test & evaluate
#读取原始测试集（不做缩放）
df = pd.read_csv(test_csv, index_col=0, parse_dates=True)
df = df[features].dropna()
timestamps = df.index
actuals = df[features[target_col]].values

#print (actuals)

results = []
max_start = len(df) - seq_len - future_hours + 1

print (max_start)
#max_start = 10
timestamps    = []
preds_first   = []
actuals_first = []

with open(log_path, 'a') as log_file:
    log_file.write("\n\n=== Begin Evaluaton ===\n")
    log_file.write("\nTimestamps                Predict  Actual\n")

    for start in range(max_start):
            log_file.write("\n")
            log_file.write("Inference Epoch " + str(start) + "\n")
            # 调用 predict.py 中的 main，返回预测值列表

            with redirect_stdout(log_file): 
                preds = predict.main(
                    start_index=start,
                    plot=None,
                    return_preds=True
                )

            pred1 = preds[0]
            ts    = df.index[start + seq_len]
            actual = df.iloc[start + seq_len][target_name]

            timestamps.append(ts)
            preds_first.append(pred1)
            actuals_first.append(actual)

df_first = pd.DataFrame({
    'timestamp':    timestamps,
    'predicted':    preds_first,
    'actual':       actuals_first
}).set_index('timestamp')

y_true = df_first['actual'].values
y_pred = df_first['predicted'].values

# 计算指标
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)


with open(log_path, 'a') as log_file:
    log_file.write(f"\n\n=== First-step Forecast vs Actual ===\n")
    log_file.write(df_first.to_string(index=True))

    log_file.write("\n\n=== Testing Loss ===\n")
    log_file.write(f"MAE:  {mae:.4f}\n")
    log_file.write(f"MSE:  {mse:.4f}\n")
    log_file.write(f"RMSE: {rmse:.4f}\n\n")

plt.figure(figsize=(12, 6))
plt.plot(df_first.index, df_first['actual'], label='Actual')
plt.plot(df_first.index, df_first['predicted'],
             linestyle='--', marker='o', label='First-step Pred')
plt.title(f'First-step Forecast vs Actual ({target_name})')
plt.xlabel('Time')
plt.ylabel(target_name)
plt.legend()
plt.tight_layout()
plt.show()

        # 对齐真实值：位置从 start+seq_len 到 start+seq_len+future_hours-1
        # for i, p in enumerate(preds):
        #     idx = start + seq_len + i
        #     results.append({
        #         'timestamp': timestamps[idx],
        #         'predicted': p,
        #         'actual':    actuals[idx]
        #     })
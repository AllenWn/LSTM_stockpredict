import yaml
import joblib
import torch
import numpy as np
import pandas as pd

from data.prepare_data import load_and_scale, make_sequences
from models.lstm_model import LSTMModel
from utils.viz import plot_prediction

# 1）加载配置 & 模型 & scaler
cfg = yaml.safe_load(open('config.yaml'))
features   = cfg['features']      # e.g. ['Close','Volume','High','Low']
target_col = cfg['target_col']    # e.g. 0 对应 'Close'
freq_map   = {'1h':'H', '1d':'D'}


# 2）拿最新数据并构造序列
df_raw = pd.read_csv(cfg['predict_data_path'], index_col=0, parse_dates=True)

if cfg.get('n_recent') is not None:
    df_raw = df_raw.tail(cfg['n_recent'])
# ensure timezone
if df_raw.index.tz is None:
    df_raw.index = df_raw.index.tz_localize('UTC').tz_convert('US/Eastern')
else:
    df_raw.index = df_raw.index.tz_convert('US/Eastern')
df_raw = df_raw[features].dropna()

scalers = joblib.load("scaler/scaler.gz")  # dict: feature -> scaler
target_scaler = scalers['Close']

df_scaled = df_raw.copy()
for col in features:
    df_scaled[col] = scalers[col].transform(df_raw[[col]])

X, y = make_sequences(df_scaled.values,
                          seq_length=cfg['seq_length'],
                          target_col=target_col)

# 取最后一个窗口，形状 (1, seq_length, num_features)
last_seq = torch.tensor(X[-1:], dtype=torch.float32)

# 4) 加载模型并切换到 eval
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = LSTMModel(
    input_size = len(features),
    hidden_size= cfg['hidden_size'],
    num_layers = cfg['num_layers'],
    output_size= 1
).to(device)
model.load_state_dict(torch.load("weights_outputs/lstm.pth", map_location=device))
model.eval()

# 5) 递推预测
preds_scaled = []
current_seq = last_seq.to(device)
with torch.no_grad():
    for _ in range(cfg['future_hours']):
        # 5.1 前向得到归一化的目标预测
        out = model(current_seq)                     # shape [1,1]
        scaled_pred = out.cpu().numpy().reshape(-1)[0]
        preds_scaled.append(scaled_pred)

        # 5.2 准备下一步输入：
        #   - 把 current_seq 最后一时刻的所有特征值取出
        last_vals = current_seq[0, -1, :].cpu().numpy()  # shape [num_features,]
        #   - 用新的预测值替换目标列对应位置
        next_vals = last_vals.copy()
        next_vals[target_col] = scaled_pred
        #   - 拼成 (1,1,num_features) 供下一次 model() 使用
        next_tensor = (
            torch.tensor(next_vals, dtype=torch.float32)
                 .unsqueeze(0)  # batch dim
                 .unsqueeze(0)  # time dim
                 .to(device)
        )
        #   - 滑动窗口：去掉最早步，将新步拼到末尾
        current_seq = torch.cat((current_seq[:,1:,:], next_tensor), dim=1)

# 6) 反归一化到真实价格
preds = target_scaler.inverse_transform(
    np.array(preds_scaled).reshape(-1,1)
).reshape(-1)

# 7) 构造未来时间索引并可视化
last_time = df_scaled.index[-1]
freq      = freq_map.get(cfg['interval'], cfg['interval'])
future_index = pd.date_range(
    start=last_time,
    periods=cfg['future_hours']+1,
    freq=freq
)[1:]

plot_prediction(
    future_index,
    preds, preds,
    title=f"Next {cfg['future_hours']}h {features[target_col]}"
)

# 8) 打印未来预测值
for t,v in zip(future_index, preds):
    print(f"{t}: {v:.2f}")


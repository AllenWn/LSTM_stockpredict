import yaml, torch, numpy as np, pandas as pd
from data.prepare_data import fetch_and_scale, make_sequences
from models.lstm_model import LSTMModel
from utils.viz import plot_prediction
import joblib

# 1）加载配置 & 模型 & scaler
cfg = yaml.safe_load(open('config.yaml'))
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = LSTMModel(2, cfg['hidden_size'], cfg['num_layers']).to(device)
model.load_state_dict(torch.load('outputs/lstm.pth', map_location=device))
model.eval()
scaler_c = joblib.load('scaler/scaler_close.gz')

# 2）拿最新数据并构造序列
df, _, _ = fetch_and_scale(cfg['ticker'], cfg['interval'], cfg['period'])
X, _ = make_sequences(df, cfg['seq_length'])

# 3）取最后一个序列，开始递推式预测
last_seq = torch.tensor(X[-1:], dtype=torch.float32).to(device)
preds = []
for _ in range(cfg['future_hours']):
    p = model(last_seq).cpu().detach().numpy().reshape(-1,1)
    price = scaler_c.inverse_transform(p)[0,0]
    preds.append(price)
    # 更新 last_seq：丢掉最旧一小时，拼上新预测

# 4）画预测结果
future_ts = pd.date_range(start=df.index[-1],
                          periods=cfg['future_hours']+1,
                          freq="H")[1:]
plot_prediction(future_ts, preds, preds,
                f"Next {cfg['future_hours']} Hours Price")

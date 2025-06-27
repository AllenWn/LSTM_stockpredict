import yaml, torch
from sklearn.model_selection import train_test_split
from data.prepare_data import fetch_and_scale, load_and_scale, make_sequences
from models.lstm_model import LSTMModel
from utils.viz import plot_loss
import torch.nn as nn, torch.optim as optim

# 1）加载配置
cfg = yaml.safe_load(open('config.yaml'))

# 2）获取并预处理数据
df, scaler_c, _ = fetch_and_scale(
    cfg['ticker'], cfg['interval'], cfg['period']
)
X, y = make_sequences(df, cfg['seq_length'])


features = ['Close','Volume','High','Low']
df, scaler = load_and_scale("data/data/AAP<_1h_3mo.csv", features)

# 3）划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4）转成 tensor 并搬到 device(CPU/MPS)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)

# 5）实例化模型、损失函数、优化器
model     = LSTMModel(input_size=2,
                      hidden_size=cfg['hidden_size'],
                      num_layers=cfg['num_layers']).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

# 6）训练循环
train_losses = []
for epoch in range(cfg['epochs']):
    model.train()           # 训练模式
    optimizer.zero_grad()   # 清零梯度
    preds = model(X_train)  # 前向传播
    loss  = criterion(preds, y_train)  # 计算损失
    loss.backward()         # 反向传播
    optimizer.step()        # 参数更新

    train_losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss={loss:.4f}")

# 7）画训练损失曲线
plot_loss(train_losses)

# 8）保存模型权重 & 保存 scaler
torch.save(model.state_dict(), 'outputs/lstm.pth')
import joblib
joblib.dump(scaler_c, 'scaler/scaler_close.gz')
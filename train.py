import yaml
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from data.prepare_data import load_and_scale, make_sequences
from models.lstm_model import LSTMModel
from utils.viz import plot_loss


cfg = yaml.safe_load(open('config.yaml'))
features   = cfg['features']      # e.g. ['Close','Volume','High','Low']
target_col = cfg['target_col']    # e.g. 0 对应 'Close'
csv_path   = cfg.get('data_path', None)

df, scalers = load_and_scale(csv_path, features = features)

scaler_target = scalers[features[target_col]]

print(df)


# X, y = make_sequences(df.values,
#                           seq_length=cfg['seq_length'],
#                           target_col=target_col)
    

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=cfg['test_size'], random_state=cfg['random_seed']
# )

   
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
    

# model = LSTMModel(
#         input_size = len(features),
#         hidden_size= cfg['hidden_size'],
#         num_layers = cfg['num_layers'],
#         output_size= 1
#     ).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    

# train_losses = []
# for epoch in range(cfg['epochs']):
#     model.train()
#     optimizer.zero_grad()
        
#     preds = model(X_train)              # 前向
#     loss  = criterion(preds, y_train)   # 计算 MSE
#     loss.backward()                     # 反向
#     optimizer.step()                    # 更新
        
#     train_losses.append(loss.item())
#     if (epoch+1) % cfg['print_every'] == 0:
#         print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss={loss.item():.6f}")
    
 

# plot_loss(train_losses)
    

# torch.save(model.state_dict(), cfg['model_output'])
# joblib.dump(scaler_target, cfg['scaler_output'])
# print(f"Model saved to {cfg['model_output']}")
# print(f"Scaler for '{features[target_col]}' saved to {cfg['scaler_output']}")
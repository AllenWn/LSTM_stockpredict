import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        # LSTM 核心层：处理序列数据
        self.lstm = nn.LSTM(
            input_size=input_size,   # 每个时间步有多少特征，这里是 2（Close,Volume）
            hidden_size=hidden_size, # LSTM 隐藏态的维度
            num_layers=num_layers,   # 堆叠几层 LSTM
            batch_first=True         # 输入/输出的第一个维度是 batch_size
        )
        # 全连接层：把最后一个时间步的隐藏状态映射到预测值
        self.fc   = nn.Linear(hidden_size, output_size)

    # x: [batch_size, seq_length, input_size]
    # 返回: [batch_size, output_size]

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
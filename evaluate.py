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
import train

cfg = yaml.safe_load(open('config.yaml'))

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

#  train()
with open(log_path, 'a') as log_file:
    log_file.write("\n=== Begin Training ===\n")
    with redirect_stdout(log_file): 
        train.main()
    log_file.write("=== End Training ===\n\n")


# test & evaluate

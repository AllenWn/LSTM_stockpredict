import matplotlib.pyplot as plt

# 画出训练过程中的损失随 epoch 变化的曲线
#     输入：train_losses 列表

def plot_loss(train_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.show()


    # 画出真实 vs 预测 的对比图
    #   - timestamps: 时间列表
    #   - actual: 真实值列表
    #   - pred:  预测值列表
    #   - title: 图表标题
    
def plot_prediction(timestamps, actual, pred, title):
    plt.figure(figsize=(10,5))
    plt.plot(timestamps, actual, label='Actual')
    plt.plot(timestamps, pred,   label='Predicted')
    plt.title(title); plt.xlabel('Time'); plt.ylabel('Price'); plt.legend(); plt.show()
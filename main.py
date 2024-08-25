import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# 載入IRIS資料集
iris = load_iris()
X, y = iris.data, iris.target

# 資料預處理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 轉換為PyTorch張量
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# 定義模型
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 交叉驗證設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

# K-Fold 交叉驗證
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    # 分割訓練集和測試集
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 初始化模型、損失函數和優化器
    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 訓練模型
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 評估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        _, predicted = torch.max(y_pred, 1)

        # 計算 accuracy
        accuracy = accuracy_score(y_test, predicted)

        # 計算 precision, recall (多類別問題，使用 micro 平均)
        precision = precision_score(y_test, predicted, average='micro')
        recall = recall_score(y_test, predicted, average='micro')

        # 計算 AUC (多類別問題，使用 OvR 策略)
        y_pred_proba = torch.softmax(y_pred, dim=1).numpy()
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        results.append((accuracy, precision, recall, auc))
        print(f'Fold {fold+1}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')

# 視覺化結果
metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
results = np.array(results)

# 為每個指標繪製圖表
for i, metric in enumerate(metrics):
    plt.figure()
    plt.plot(results[:, i], label=metric, marker='o')
    plt.title(f'{metric} Across Folds')
    plt.xlabel('Fold Number')
    plt.ylabel(metric)
    plt.xticks(range(len(results)), [f'Fold {j+1}' for j in range(len(results))])
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./Iris_Classification_{metric}.png')
    plt.show()

# 輸出平均結果
avg_results = np.mean(results, axis=0)
print(f'Average Accuracy: {avg_results[0]:.4f}')
print(f'Average Precision: {avg_results[1]:.4f}')
print(f'Average Recall: {avg_results[2]:.4f}')
print(f'Average AUC: {avg_results[3]:.4f}')

# 說明

## 本次實驗使用的資料集

本次實驗使用的資料集為IRIS，並且使用交叉驗證。

### 模型架構
這次實驗使用的模型為一簡單的線性模型，架構如下:
```
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 使用教學
```
python main.py
```
### 訓練過程:
```
Fold 1, Accuracy: 0.9667, Precision: 0.9667, Recall: 0.9667, AUC: 1.0000
Fold 2, Accuracy: 0.9667, Precision: 0.9667, Recall: 0.9667, AUC: 1.0000
Fold 3, Accuracy: 0.9333, Precision: 0.9333, Recall: 0.9333, AUC: 1.0000
Fold 4, Accuracy: 0.9667, Precision: 0.9667, Recall: 0.9667, AUC: 1.0000
Fold 5, Accuracy: 0.9667, Precision: 0.9667, Recall: 0.9667, AUC: 0.9969
-------------------------------------
Average Accuracy: 0.9600
Average Precision: 0.9600
Average Recall: 0.9600
Average AUC: 0.9994
```
### 訓練結果:

Accuracy :
![training_result](https://imgur.com/EixGAV5.jpg)

Precision :
![training_performance](https://imgur.com/UgoX0pO.jpg)

Recall :
![matrix](https://imgur.com/66vnUHq.jpg)

AUC :
![training_performance](https://imgur.com/ZDSOcta.jpg)

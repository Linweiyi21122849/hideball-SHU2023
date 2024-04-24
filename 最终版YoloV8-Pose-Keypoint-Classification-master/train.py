import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#将scv文件读取并存储在 DataFrame 对象 df 中
# arg=============================================== #

# df = pd.read_csv('datasets\Legspose_keypoint.csv')
df = pd.read_csv('datasets\Handspose_keypoint.csv')

# X = df.iloc[:,24:] # 24: 表示训练下肢，用来判断是否站立 12
X = df.iloc[:,12:24]  # 8:24 表示训练上肢，用来判断手部动作 16 
# print(X.columns)
# print(X.shape[1])

# PATH_SAVE = 'models\Legspose_classification.pt'# 表示训练下肢
PATH_SAVE = 'models\Handspose_classification.pt'#表示训练上肢

num_epoch = 100
# ================================================== #

# encoder label创建了一个 LabelEncoder 对象，用于将标签进行编码
encoder = LabelEncoder()
# 这行代码从数据框 df 中选择了名为 'label' 的列，并将其赋值给变量 y_label。
y_label = df['label']
#将原始的标签值映射到一个整数编码的数组 y 中。
y = encoder.fit_transform(y_label)
#计算了每个类别的权重
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)


# 分离出训练级和测试级
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=2022)
print("Number of Training keypoints: ", len(X_train))
print("Number of Testing keypoints: ", len(X_test))
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class DataKeypointClassification(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
train_dataset = DataKeypointClassification(X_train, y_train)
test_dataset = DataKeypointClassification(X_test, y_test)
batch_size = 12
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
hidden_size = 256
model = NeuralNet(X_train.shape[1], hidden_size, len(class_weights))
len(class_weights)
#定义了学习率、损失函数和优化器
learning_rate = 0.01
criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights.astype(np.float32)))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    train_acc = 0
    train_loss = 0
    loop = tqdm(train_loader)
    for idx, (features, labels) in enumerate(loop):
        outputs = model(features)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == labels).sum().item()
        accuracy = correct / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch}/{num_epoch}]")
        loop.set_postfix(loss=loss.item(), acc=accuracy)

test_features = torch.from_numpy(X_test.astype(np.float32))
test_labels = y_test
with torch.no_grad():
    outputs = model(test_features)
    _, predictions = torch.max(outputs, 1)
predictions

print(classification_report(test_labels, predictions, target_names=encoder.classes_))

torch.save(model.state_dict(), PATH_SAVE)
import math
import torch.nn.functional as F
import scipy.optimize as sopt

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 加载数据
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('All_data_Male3638.csv')

# 合并所有的数据集
all_data = pd.concat([train_df, valid_df, test_df])

# 对所有数据进行预处理和独热编码
all_features_df = pd.get_dummies(all_data[['SEX', 'AGE', 'INITIAL_EVIDENCE']], columns=['SEX', 'INITIAL_EVIDENCE'])

# 再分割为训练、验证和测试集
train_len = len(train_df)
valid_len = len(valid_df)
train_features_df = all_features_df[:train_len]
valid_features_df = all_features_df[train_len:train_len+valid_len]
test_features_df = all_features_df[train_len+valid_len:]

# 使用LabelEncoder进行标签编码
le = LabelEncoder()
train_target_df = le.fit_transform(train_df['PATHOLOGY'])
valid_target_df = le.transform(valid_df['PATHOLOGY'])
test_target_df = le.transform(test_df['PATHOLOGY'])

# 将数据转换为tensor
train_features = torch.tensor(train_features_df.values, dtype=torch.float32)
train_targets = torch.tensor(train_target_df, dtype=torch.long)
valid_features = torch.tensor(valid_features_df.values, dtype=torch.float32)
valid_targets = torch.tensor(valid_target_df, dtype=torch.long)
test_features = torch.tensor(test_features_df.values, dtype=torch.float32)
test_targets = torch.tensor(test_target_df, dtype=torch.long)

# 创建DataLoader
batch_size = 32
train_data = TensorDataset(train_features, train_targets)
train_loader = DataLoader(train_data, batch_size=batch_size)

valid_data = TensorDataset(valid_features, valid_targets)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

test_data = TensorDataset(test_features, test_targets)
test_loader = DataLoader(test_data, batch_size=batch_size)

# 创建模型
model = nn.Sequential(
    nn.Linear(train_features_df.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, len(le.classes_))
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters())

max_l = 10.
alpha = 0.2
C = math.sqrt(1 + (1 / alpha - 1) ** 2)

# 训练模型
model.train()
for epoch in range(50):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        foo = lambda eta: C * math.sqrt((F.relu(losses - eta) ** 2).mean().item()) + eta
        opt_eta = sopt.brent(foo, brack=(0, max_l))
        losses = C * torch.sqrt((F.relu(losses - opt_eta) ** 2).mean()) + opt_eta
        losses.backward()
        optimizer.step()

# 测试模型
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).mean().item()  # 将一批的损失相加
        pred = outputs.argmax(dim=1, keepdim=True)  # 获得概率最高的预测
        correct += pred.eq(targets.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
print('Test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


# 0) Prepare data
bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target

n_samples, n_features = x.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 15)

# scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Model
# Logistic regression = Linear Regression + Sigmoid(sonucu 0 ile 1 arasına sıkıştırır)
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_features)

# 2) Loss and optimizer
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCELoss() # Logistic Regressionda BCE loss u kullanılır
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    #Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before and update
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}, loss {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    acc = (y_predicted_cls.eq(y_test).sum() / y_test.shape[0])
    print(f'accuracy {acc.item():.4f}')
'''
1️⃣ Gradient kapat
2️⃣ Test verisiyle tahmin al
3️⃣ Olasılığı 0/1 yap(round ile yuvarlar)
4️⃣ Gerçek değerle karşılaştır
5️⃣ Doğru sayısını bul
6️⃣ Accuracy hesapla
7️⃣ Yazdır(.item tensor u float a çevirir)
'''

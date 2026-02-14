# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch
import torch.nn as nn

from MyWorking.Pytorch.GradientsTorch import learning_rate

# Linear regression
# f = w * x

# here : f = 2 * x

# 0) Training samples
X = torch.tensor([1,2,3,4], dtype = torch.float32)
Y = torch.tensor([2,4,6,8], dtype = torch.float32)

# 1) Desing Model: Weights to optimize and forward function
w = torch.tensor(0.0, dtype=torch.float32, requires_grad = True)

def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

# callable function
loss = nn.MSELoss()

optimizer = torch.optim.SGD([w], lr = learning_rate)

# 3) Training loop
for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    #loss
    l=loss(Y, y_pred)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print('epoch ', epoch+1, ': w = ',w,' loss = ',l)

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

"""
### 1) GENEL AKIŞ (Pytorch Pipeline)
- Veri (X, Y): Modelin öğrenmesi gereken ilişkiyi (Y = 2X) tanımladık.
- Model: w * x şeklinde basit bir ileri besleme (forward) fonksiyonu yazdık.
- Loss: Tahmin ile gerçek arasındaki farkı MSE (Mean Squared Error) ile ölçtük.
- Optimizer: Hataları düzeltmek için SGD kullandık.

### 2) NEDEN SGD (Stochastic Gradient Descent)?
- Hocanın SGD kullandırmasının sebebi, türevin (gradyanın) ağırlığı nasıl 
  doğrudan etkilediğini en çıplak haliyle göstermektir. 
- SGD'de kural nettir: w = w - (lr * gradient). 
- Arada hiçbir akıllı filtre olmadığı için matematiği elle takip etmek kolaydır.

### 3) ADAM VS. SGD (Gradyan Meselesi)
- "Neden Adam kullanınca da gradyanlara bakıyoruz?" sorusunun cevabı: 
  İki algoritma da 'l.backward()' ile gradyanı hesaplar; ama kullanma şekilleri farklıdır.
- SGD: Gradyanı olduğu gibi alır, yönü nereye gösteriyorsa oraya gider.
- ADAM: Gradyanı alır ama onu 'Momentum' ve 'Adaptive Learning Rate' süzgeçlerinden geçirir. 
  Yani gradyanı ham haliyle değil, geçmişteki hataları da hesaba katarak "optimize edilmiş" 
  şekilde kullanır.

### SONUÇ:
Gradyanlara bakmayacak kadar profesyonelleştiğimizde Adam daha iyidir (daha hızlı yakınsar). 
Ancak işin mantığını (backpropagation) anlamak için SGD 'altın standarttır'.
"""
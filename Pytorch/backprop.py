import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2
print(loss)

#backward pass
loss.backward()
print(w.grad)

# update weights
# next forward and backwars

with torch.no_grad():
    w -= 0.01 * w.grad

w.grad.zero_()

# y_hat = w * x: İleri besleme (forward pass) adımıdır; modelin şu anki ağırlığıyla yaptığı tahmindir.
# loss = (y_hat - y) ** 2: Hata payını hesaplar; tahmin ile gerçek arasındaki farkın karesini alırız.
# loss.backward(): Geriye yayılım (backward pass) adımıdır; hatayı azaltmak için hangi ağırlığın ne yöne değişeceğini (türevini) hesaplar.
# w.grad: backward() komutuyla hesaplanan türev (gradyan) değerinin saklandığı yerdir.
# update weights: Hesaplanan bu gradyanı kullanarak ağırlığı güncellediğimiz (öğrenmenin gerçekleştiği) yerdir.
# next forward and backwards: Gradyanları sıfırlayıp (zero_grad) döngüyü baştan başlatarak hatayı iyice azaltma sürecidir.
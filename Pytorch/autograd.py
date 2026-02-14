import torch

x = torch.rand(3, requires_grad=True)

with torch.no_grad():
    y_hidden = x + 2

y_detached = (x * 2).detach()

y = x + 2
z = y * y * 2

v = torch.ones(3)
z.backward(v)

print(x.grad)

# requires_grad=True: PyTorch'a bu değişkenin türevini (gradyanını) hesaplaması gerektiğini söyler.
# with torch.no_grad(): Bu blok içindeki işlemleri hafızada tutmaz ve türev hesaplamaz (hız ve bellek tasarrufu).
# .detach(): Bir tensörü bağlı olduğu hesaplama grafiğinden tamamen koparır, geçmişini siler.
# z.backward(v): Zincir kuralını (Chain Rule) kullanarak sondan başa doğru türevleri hesaplar.
# x.grad: backward() sonucu oluşan türev değerlerinin saklandığı yerdir.
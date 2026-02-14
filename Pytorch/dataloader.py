import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


# 1. KENDİ VERİ SETİMİZİ OLUŞTURUYORUZ
# PyTorch'un veriyi anlaması için Dataset sınıfından bir 'çocuk' (subclass) türetiriz.
class WineDataset(Dataset):
    def __init__(self):
        # Bu kısım 'hazırlık' aşamasıdır. Veri dosyasını açar ve belleğe yükleriz.
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)

        # Toplam kaç satır (örnek) olduğunu kaydedelim.
        self.n_samples = xy.shape[0]

        # Veriyi parçalara ayırıyoruz:
        # x_data: Özellikler (alkol oranı, asitlik vb.) - İlk sütun hariç her şey
        self.x_data = torch.from_numpy(xy[:, 1:])
        # y_data: Hedef/Etiket (şarabın türü) - Sadece ilk sütun
        self.y_data = torch.from_numpy(xy[:, [0]])

    def __getitem__(self, index):
        # Bu metod 'liste' gibi çalışmamızı sağlar.
        # dataset[5] dediğimizde bize 5. sıradaki veriyi ve onun etiketini getirir.
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # Veri setinde toplam kaç tane örnek olduğunu söyler.
        return self.n_samples


if __name__ == '__main__':
    # 2. VERİ YÜKLEYİCİYİ (DATALOADER) AYARLIYORUZ
    dataset = WineDataset()

    # DataLoader, veriyi 'batch' (lokma) halinde sunan mekanizmadır.
    train_loader = DataLoader(
        dataset=dataset,  # Hangi veri setini kullanacak?
        batch_size=4,  # Her seferinde kaç örnek getirsin? (4'er 4'er)
        shuffle=True,  # Her seferinde veriyi karıştırsın mı? (Evet, ezberlemesin diye)
        num_workers=2  # Veriyi hazırlarken kaç tane işlemci çekirdeği çalışsın?
    )

    # 3. EĞİTİM DÖNGÜSÜ (TEMSİLİ)
    num_epochs = 2  # Tüm verinin üzerinden toplam kaç kez geçilecek?

    for epoch in range(num_epochs):
        # 'enumerate' sayesinde kaçıncı pakette (step) olduğumuzu biliriz.
        for i, (inputs, labels) in enumerate(train_loader):
            # Burada yapay zeka eğitiliyor...
            # inputs: 4 adet şarap özelliği
            # labels: o 4 şarabın gerçek türü
            if (i + 1) % 5 == 0:
                print(f'Tur: {epoch + 1}, Adım: {i + 1} | Veri Paketi Boyutu: {inputs.shape}')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. HAZIR VERİ SETİNİ İNDİRME VE AYARLAMA
# torchvision.datasets içinde birçok popüler veri seti (MNIST, CIFAR10 vb.) hazır bulunur.
train_dataset = torchvision.datasets.MNIST(
    root='./data',              # Verinin bilgisayarında nereye kaydedileceği
    train=True,                 # Eğitim verilerini mi (True) yoksa test verilerini mi (False) istiyorsun?
    transform=torchvision.transforms.ToTensor(), # EN ÖNEMLİSİ: Resimleri PyTorch'un anlayacağı 'Tensor' formatına çevirir
    download=True               # Eğer klasörde veri yoksa internetten otomatik indirir
)

# 2. VERİ YÜKLEYİCİYİ (DATALOADER) OLUŞTURMA
# Veri seti indikten sonra, onu yine küçük paketler (batch) halinde kullanmamız gerekir.
train_loader = DataLoader(
    dataset=train_dataset,      # Yukarıda indirdiğimiz MNIST veri setini kullan
    batch_size=3,               # Her seferinde 3'er 3'er resim getir (Genelde 32, 64 veya 128 kullanılır)
    shuffle=True                # Modelin sırayı ezberlememesi için her seferinde karıştır
)

# 3. VERİLERİ KONTROL ETME (GÖRÜNTÜLEME)
# DataLoader bir 'üretici' (iterator) gibidir. İçinden bir paket veri çekip bakalım:
dataiter = iter(train_loader)
inputs, targets = next(dataiter)

# Ekrana yazdırıp boyutlarına bakalım:
print(f"Giriş (Resimler) Boyutu: {inputs.shape}")
print(f"Hedef (Etiketler) Boyutu: {targets.shape}")

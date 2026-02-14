import torch
import torch.nn as nn


# --- KONU: BINARY CLASSIFICATION (İKİLİ SINIFLANDIRMA) ---
# Kullanım Alanı: Spam/Normal, Hasta/Sağlıklı, Kedi/Kedi Değil
# Kritik Fonksiyon: nn.BCEWithLogitsLoss (İçinde Sigmoid barındırır)

def binary_class_ornegi():
    # 1. Hazırlık
    # Model çıktısı (Batch size: 2). Rastgele ham puanlar.
    outputs = torch.randn(2, 1, requires_grad=True)

    # Gerçek Etiketler (Labels)
    # DİKKAT: Binary'de etiketler FLOAT tipinde olmalı (1.0 veya 0.0)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    # 2. Loss Fonksiyonunu Tanımla
    # Sigmoid + BCELoss işlemini tek seferde yapar. Daha stabildir.
    criterion = nn.BCEWithLogitsLoss()

    # 3. Hatayı Hesapla
    loss = criterion(outputs, targets)

    print("--- Binary Classification ---")
    print(f"Model Çıktıları (Raw): \n{outputs.detach()}")
    print(f"Hedefler: \n{targets}")
    print(f"Hesaplanan Loss: {loss.item():.4f}")


if __name__ == "__main__":
    binary_class_ornegi()
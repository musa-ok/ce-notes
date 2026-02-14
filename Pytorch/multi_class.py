import torch
import torch.nn as nn


# --- KONU: MULTICLASS CLASSIFICATION (ÇOKLU SINIFLANDIRMA) ---
# Kullanım Alanı: 0'dan 9'a rakam tanıma, Kedi/Köpek/Kuş/At sınıflandırma
# Kritik Fonksiyon: nn.CrossEntropyLoss (İçinde Softmax barındırır)

def multi_class_ornegi():
    # 1. Hazırlık
    # Batch size: 2, Sınıf Sayısı: 3
    # Model çıktısı ham puanlardır (Logits). Softmax YOK!
    outputs = torch.randn(2, 3, requires_grad=True)

    # Gerçek Etiketler (Class Indices)
    # DİKKAT: Multiclass'ta etiketler LONG (tamsayı) olmalı ve sınıfın indexini vermeli.
    # 1. resim -> Sınıf 0, 2. resim -> Sınıf 2 olsun.
    targets = torch.tensor([0, 2], dtype=torch.long)

    # 2. Loss Fonksiyonunu Tanımla
    # Softmax + NLLLoss işlemini tek seferde yapar.
    criterion = nn.CrossEntropyLoss()

    # 3. Hatayı Hesapla
    loss = criterion(outputs, targets)

    print("--- Multiclass Classification ---")
    print(f"Model Çıktıları (Raw): \n{outputs.detach()}")
    print(f"Doğru Sınıf Indexleri: {targets}")
    print(f"Hesaplanan Loss: {loss.item():.4f}")

    # EKSTRA: Tahminleri Görmek İstersek (Softmax ekleriz)
    probs = nn.Softmax(dim=1)(outputs)
    print(f"\nGerçek Olasılıklar (Tahmin): \n{probs.detach()}")


if __name__ == "__main__":
    multi_class_ornegi()
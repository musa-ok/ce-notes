import torch
import torch.nn as nn


# --- KONU: CROSS ENTROPY MANTIĞI ---
# Mantık: Modelin tahmin ettiği olasılık ile gerçek cevap arasındaki mesafeyi ölçer.
# Kural: Model doğru cevaba ne kadar düşük olasılık verirse, CE cezayı (Loss) o kadar artırır.

def cross_entropy_manuel():
    # SENARYO: 3 Sınıf var (0:Kedi, 1:Köpek, 2:Kuş)
    # Gerçek cevap: Kedi (Index 0)

    # 1. Modelin Tahminleri (Softmax'tan çıkmış olasılıklar)
    # Senaryo A: İyi Tahmin (Kediye %70 vermiş)
    probs_iyi = torch.tensor([0.7, 0.2, 0.1])

    # Senaryo B: Kötü Tahmin (Kediye %10 vermiş)
    probs_kotu = torch.tensor([0.1, 0.7, 0.2])

    # 2. Manuel Hesap (Sadece doğru sınıfın -logaritması)
    # Loss = -Log(Doğru Sınıfın Olasılığı)
    loss_iyi = -torch.log(probs_iyi[0])
    loss_kotu = -torch.log(probs_kotu[0])

    print(f"İyi Tahmin Kaybı (Loss): {loss_iyi.item():.4f}")  # Düşük çıkmalı
    print(f"Kötü Tahmin Kaybı (Loss): {loss_kotu.item():.4f}")  # Yüksek çıkmalı


if __name__ == "__main__":
    cross_entropy_manuel()
import torch
import torch.nn as nn

# --- KONU: SOFTMAX ---
# Mantık: Eksi/Artı rastgele sayıları (Logits) alır,
# toplamı 1 olan "Olasılık Dağılımına" çevirir.
# Soru: "Bu resim yüzde kaç ihtimalle Kedi?" cevabını verir.

def softmax_ornegi():
    # 1. Modelden çıkan ham puanlar (Logits)
    # [Kedi Puanı, Köpek Puanı, Kuş Puanı]
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    print(f"Ham Puanlar (Logits): {logits}")

    # 2. Softmax Uygulama (dim=1 -> Satır bazında işlem yap)
    softmax = nn.Softmax(dim=1)
    probs = softmax(logits)

    # 3. Sonuç
    print(f"Olasılıklar: {probs}")
    print(f"Olasılık Toplamı: {probs.sum().item()}") # 1.0 çıkmalı

if __name__ == "__main__":
    softmax_ornegi()
# Başlangıç ayarları
best_loss = float('inf')  # En iyi hatayı sonsuz kabul ediyoruz
patience = 5  # "Sabır" süresi: Hata düşmese bile kaç epoch daha bekleyelim?
counter = 0  # Hatanın düşmediği epochları sayan sayaç

epochs = 500
for epoch in range(epochs):
    model.train()
    # ... (Eğitim kodları: y_pred, loss, backward, step) ...

    # --- EARLY STOPPING KONTROLÜ ---
    model.eval()
    with torch.no_grad():
        val_pred = model(x_test)
        val_loss = criterion(val_pred, y_test)

    # Eğer bu epoch'taki hata, şimdiye kadarki en iyi hatadan düşükse:
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0  # Sayacı sıfırla, doğru yoldayız!
        # En iyi modeli kaydetmek için ideal yer burasıdır
        torch.save(model.state_dict(), 'en_iyi_model.pth')
    else:
        counter += 1  # Hata düşmedi, sabrımız azalıyor...

    if counter >= patience:
        print(f"Kanka {epoch + 1}. epochta durdum! Çünkü {patience} turdur gelişme yok.")
        break  # Döngüden çık, daha fazla ezberlemesin!

# Lazımsa kullanabilirsin eğer model loss u artıyorsa takip edemiyorsan
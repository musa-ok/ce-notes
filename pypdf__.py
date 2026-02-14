from pypdf import PdfReader

okuyucu = PdfReader("../AskMyDocs/Deneme1.pdf")

sayfa_sayisi = len(okuyucu.pages)
print(f"Sayfa sayısı: {sayfa_sayisi}")

ilk_sayfa = okuyucu.pages[0]
metin = ilk_sayfa.extract_text()

print(metin)
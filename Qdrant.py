[28.01.2026 16:28:53] Musa: from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ==========================================
# 1. ADIM: DÜKKANI AÇ (Bağlantı)
# ==========================================
# ":memory:" parametresi çok önemli.
# Bunu yazınca bilgisayarına dosya kaydetmez, her şeyi RAM'de tutar.
# Kod bitince her şey silinir. Öğrenci dostudur, çöplük yaratmaz.
client = QdrantClient(location=":memory:")

print("✅ Qdrant RAM üzerinde başlatıldı!")

# ==========================================
# 2. ADIM: RAF SİSTEMİNİ KUR (Collection)
# ==========================================
# collection_name: Tablo adı gibi düşün. "belgelerim", "filmler" vs.
# vectors_config: Burası kritik.
#   - size: 4 -> Her bir veriyi 4 tane sayıyla ifade edeceğiz dedik.
#             (Gerçek projede bu sayı 384, 768 veya 1536 olur).
#   - distance: Benzerliği neyle ölçeceğiz? Cosine (Açı) genelde standarttır.
client.create_collection(
    collection_name="deneme_tahtasi",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)

print("✅ 'deneme_tahtasi' adında raf oluşturuldu!")

# ==========================================
# 3. ADIM: VERİLERİ YÜKLE (Upsert)
# ==========================================
# Upsert = Update + Insert (Varsa güncelle, yoksa ekle)
# PointStruct: Veritabanına girecek her bir satır "Point"tir.
#   - id: Mecburidir. Sayı (1, 2) veya UUID olabilir.
#   - vector: Verinin matematiksel karşılığı. (Yukarıda size=4 dedik, buraya 4 sayı girmeliyiz)
#   - payload: Verinin asıl kendisi (Metin, Başlık, Sayfa No vs.)
operasyon = client.upsert(
    collection_name="deneme_tahtasi",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.9, 0.1, 0.0], # Temsili: "Futbol" kelimesinin sayıları olsun
            payload={"kategori": "Spor", "metin": "Galatasaray şampiyon oldu."}
        ),
        PointStruct(
            id=2,
            vector=[0.8, 0.1, 0.0, 0.1], # Temsili: "Yazılım" kelimesinin sayıları olsun
            payload={"kategori": "Teknoloji", "metin": "Python öğrenmek çok zevkli."}
        ),
        PointStruct(
            id=3,
            vector=[0.1, 0.8, 0.2, 0.0], # Temsili: "Basketbol" (Futbola benziyor sayıları)
            payload={"kategori": "Spor", "metin": "NBA finalleri başladı."}
        )
    ]
)

print(f"✅ Veriler yüklendi. Durum: {operasyon.status}")

# ==========================================
# 4. ADIM: ARAMA YAP (Search)
# ==========================================
# Şimdi soruyoruz: [0.1, 0.9, 0.1, 0.0] vektörüne (Futbol) en çok benzeyen kim?
# query_vector: Aradığımız şeyin sayısal hali.
# limit: Kaç tane sonuç getirsin? (En benzer 2 taneyi getir dedik)
sonuclar = client.search(
    collection_name="deneme_tahtasi",
    query_vector=[0.1, 0.9, 0.1, 0.0],
    limit=2
)

print("\n🔍 --- ARAMA SONUÇLARI ---")
for sonuc in sonuclar:
    # score: Benzerlik puanı (1.0 demek birebir aynısı demek)
    print(f"Benzerlik Puanı: {sonuc.score:.4f}")
    print(f"İçerik: {sonuc.payload['metin']}")
    print("---")
[28.01.2026 16:29:05] Musa: Qdrant
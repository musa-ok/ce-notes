from sentence_transformers import SentenceTransformers

model = SentenceTransformer('all-MiniLM-L6-v2')

cumle = "Bu bir deneme cümlesidir."
sayilar = model.encode(cumle)

print("----Metnin Sayısal Hali----")
print(sayilar)
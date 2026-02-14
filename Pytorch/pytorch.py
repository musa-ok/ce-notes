import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = torch.ones(1, 3, 5, 5, device=device)

# torch.device: İşlemlerin hangi donanım birimi (CPU, NVIDIA GPU/CUDA veya Mac GPU/MPS) üzerinde yapılacağını belirler.
# torch.backends.mps.is_available():  Mac cihazındaki GPU (Metal) hızlandırmasının aktif olup olmadığını kontrol eder.
# device=device: Tensörün oluşturulurken doğrudan belirlenen donanım hafızasına (örneğin işlemci yerine GPU'ya) yazılmasını sağlar.
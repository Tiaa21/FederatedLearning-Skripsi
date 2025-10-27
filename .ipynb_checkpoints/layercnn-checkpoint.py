import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from networks import Classifier  # pastikan file networks.py ada di path kamu

# --- 1. Siapkan model ---
model = Classifier()
encoder = model.encoder.view_resnet  # ambil bagian encoder (ResNet22)
model.eval()

# --- 2. Ambil beberapa layer conv untuk divisualisasi ---
layer_outputs = {}
def get_activation(name):
    def hook(model, input, output):
        layer_outputs[name] = output.detach()
    return hook

# Daftar layer yang mau kamu lihat (bisa disesuaikan)
target_layers = {
    "Conv1": encoder.first_conv,
    "Layer1": encoder.layer_list[0],
    "Layer2": encoder.layer_list[1],
    "Layer3": encoder.layer_list[2],
    "Layer4": encoder.layer_list[3],
    "Layer5": encoder.layer_list[4],
}

for name, layer in target_layers.items():
    layer.register_forward_hook(get_activation(name))

# --- 3. Input gambar dummy (grayscale) ---
# Ganti dengan gambar kamu sendiri kalau mau
x = torch.randn(1, 1, 256, 256)

# Jalankan forward
_ = encoder(x)

# --- 4. Visualisasi hasil tiap layer ---
fig, axes = plt.subplots(1, len(target_layers)+1, figsize=(20, 6))
axes[0].imshow(x[0,0].cpu(), cmap='gray')
axes[0].set_title("Input", fontsize=12)
axes[0].axis('off')

for idx, (name, feat) in enumerate(layer_outputs.items()):
    # Ambil 1 channel tengah untuk ditampilkan
    fmap = feat[0, feat.shape[1] // 2].cpu()
    axes[idx+1].imshow(fmap, cmap='viridis')
    axes[idx+1].set_title(name, fontsize=12)
    axes[idx+1].axis('off')

plt.tight_layout()
plt.show()
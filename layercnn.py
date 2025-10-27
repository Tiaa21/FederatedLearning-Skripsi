import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from networks import Classifier  # pastikan networks.py di folder yang sama

# --- 1. Load model ---
model = Classifier()
encoder = model.encoder.view_resnet  # ambil bagian encoder (ResNet22)
model.eval()

# --- 2. Load gambar asli kamu ---
image_path = r"C:\Users\Chyntia Irawan\Documents\GitHub\curriculum-federated-learning\dataset_cbis\site0\train\benign\1-253.jpg"

img = Image.open(image_path).convert("L")  # ubah ke grayscale
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # sesuaikan ukuran input model
    transforms.ToTensor(),
])
x = transform(img).unsqueeze(0)  # tambahkan batch dimension: (1, 1, 1024, 1024)

# --- 3. Hook untuk ambil feature map tiap layer ---
layer_outputs = {}
def get_activation(name):
    def hook(model, input, output):
        layer_outputs[name] = output.detach()
    return hook

target_layers = {
    "Conv1": encoder.first_conv,
    "Layer1": encoder.layer_list[0],
    "Layer2": encoder.layer_list[1],
    "Layer3": encoder.layer_list[2],
    "Layer4": encoder.layer_list[3],
    "Layer5": encoder.layer_list[4],
    "Final_BN": encoder.final_bn,
    "Final_ReLU": encoder.activation
}

for name, layer in target_layers.items():
    layer.register_forward_hook(get_activation(name))

# --- 4. Jalankan forward pass ---
with torch.no_grad():
    _ = encoder(x)

# --- 5. Visualisasi hasil tiap layer ---
fig, axes = plt.subplots(1, len(target_layers)+1, figsize=(22, 8))
axes[0].imshow(x[0, 0].cpu(), cmap='gray')
axes[0].set_title("Input", fontsize=12)
axes[0].axis('off')

import math

num_filters_to_show = 6  # tampilkan 6 feature map per layer

for name, feat in layer_outputs.items():
    num_filters = min(num_filters_to_show, feat.shape[1])
    cols = num_filters
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3))
    fig.suptitle(name, fontsize=14)

    for i in range(num_filters):
        ax = axes if num_filters == 1 else axes[i]
        fmap = feat[0, i].cpu().numpy()
        ax.imshow(fmap, cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
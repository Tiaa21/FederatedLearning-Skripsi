import os
import torchvision.transforms as transforms
import torch
from torchvision import transforms
from PIL import ImageFilter

class Sharpen:
    def __call__(self, img):
        return img.filter(ImageFilter.SHARPEN)

# --- PHE params ---
# mode = "Plain"
mode = "CKKS"
ckks_nslots = 1024

# model hyperparameters
pretrained = True  # use pretrained weights for feature extractor

# federated learning
nsteps = 120  # 60
pace = 40  # 20
noise_type = 'G'
# noise = 0.001
noise = 0
n_epochs_adversarial = 10  # start propagating adversarial loss for domain adaptation after "X" epochs
torch_seed = 0
n_sites = 3
use_curriculum = True
patience = 15

# optimization hyperparameters
n_epochs = 100  # number of epochs
batch_size = 4  # batch size
learning_rate = 1E-5  # learning rate
weight_decay = 1E-4  # weight decay
optimizer = 'AdamW'   # optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data parameters
preprocess = True # apply preprocessing to images
data_seed = 42  # seed for train/val split
num_workers = 0
ignore_label = None   # 'benign'   # train normal / cancer
n_classes = 2  # number of classes
input_size = 2048  # resize images to input_size pixels

# transformations to apply to the data
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((input_size, input_size)),
    transforms.ColorJitter(brightness=0.2, contrast=0.4),
    Sharpen(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])    # -> [-1,1]
])

data_path = os.path.join(os.getcwd(), 'dataset_cbis')

dpath = {
    f"site{site}": {
        "train": os.path.join(data_path, f"site{site}", "train"),
        "val":   os.path.join(data_path, f"site{site}", "val"),
        "test":  os.path.join(data_path, f"site{site}", "test"),
    }
    for site in range(3)
}

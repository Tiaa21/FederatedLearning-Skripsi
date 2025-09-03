import os
import torchvision.transforms as transforms
import torch


# model hyperparameters
pretrained = True  # use pretrained weights for feature extractor

# federated learning
nsteps = 120  # 60
pace = 40  # 20
noise_type = 'G'
noise = 0.001
n_epochs_adversarial = 5  # start propagating adversarial loss for domain adaptation after "X" epochs
torch_seed = 0
n_sites = 3

# optimization hyperparameters
n_epochs = 51  # number of epochs
batch_size = 4  # batch size
learning_rate = 1E-5  # learning rate
weight_decay = 1E-4  # weight decay
optimizer = 'adam'   # optimizer

# data parameters
preprocess = False  # apply preprocessing to images
data_seed = 42  # seed for train/val split
num_workers = 0
ignore_label = None   # 'benign'   # train normal / cancer
n_classes = 2  # number of classes
input_size = 2048  # resize images to input_size pixels

# transformations to apply to the data
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),                 # -> [0,1]
    transforms.Normalize([0.5], [0.5])     # -> [-1,1] aman & konsisten
])

data_path = os.path.join(os.getcwd(), 'dataset_inbreast')

dpath = dict()
for site in range(3):
    dpath[f"site{site}"] = {
        "train": os.path.join(data_path, f"site{site}", "train"),
        "val":   os.path.join(data_path, f"site{site}", "val"),
        "test":  os.path.join(data_path, f"site{site}", "test"),
    }

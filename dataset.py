import torch
from torch.utils.data import Dataset, Subset, Sampler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import params
from skimage import io
from torch.utils.data import Subset
from torchvision import datasets
from torch.utils.data import DataLoader

class ImageFolderWithDomain(datasets.ImageFolder):
    def __init__(self, root, transform=None, domain_label=0):
        super().__init__(root, transform=transform)
        self.domain_label = int(domain_label)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)          # (PIL -> transform -> tensor), label 0/1
        return img, label, self.domain_label, index

def get_filenames(main_dir, ignore_label):
    image_paths = []
    for mydir in main_dir:
        # r=root, d=directories, f = files
        for r, d, f in os.walk(mydir):
            for file in f:
                if file.endswith(".png"):
                    if ignore_label == 'normal':
                        if 'normal' in os.path.join(r, file) or 'cls-b1' in os.path.join(r, file):
                            continue
                    elif ignore_label == 'benign':
                        if 'benign' in os.path.join(r, file) or 'cls-b2' in os.path.join(r, file):
                            continue
                    filename = os.path.join(r, file)
                    image_paths.append(filename)
    return sorted(image_paths)

def get_loaders(site, batch_size, transform, num_workers=0):
    paths = params.dpath[f"site{site}"]

    train_ds = ImageFolderWithDomain(paths["train"], transform=transform, domain_label=site)
    val_ds   = ImageFolderWithDomain(paths["val"],   transform=transform, domain_label=site)
    test_ds  = ImageFolderWithDomain(paths["test"],  transform=transform, domain_label=site)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_class(filename):
    # if 'normal' in filename or 'cls-b1' in filename:
    if 'normal' in filename or 'benign' in filename:
        label = 0
    elif 'malignant' in filename:
        label = 1
    # elif 'benign' in filename:
    #     label = 2
    else:
        print('error: unknown class')
    return label


def get_domain(filename):
    if 'inbreast' in filename:
        label = 0
    elif 'hologic' in filename:
        label = 1
    elif 'ge' in filename:
        label = 2
    elif 'ddsm' in filename:
        label = 3
    elif 'fujifilm' in filename:
        label = 4
    elif 'planmed' in filename:
        label = 5
    else:
        print('error: unknown domain')
    return label


def _get_all_labels(main_dir):
    labels = []
    for filename in main_dir:   
        labels.append(get_class(filename))
    return np.asarray(labels)


def _get_all_domains(main_dir):
    domains = []
    for filename in main_dir:
        domains.append(get_domain(filename))
    return np.asarray(domains)


def _normalize(arr):
    ''' Function to scale an input array to [-1, 1] '''
    arr_min = arr.min()
    arr_max = arr.max()
    # Check the original min and max values
    #print('Min: %.3f, Max: %.3f' % (arr_min, arr_max))
    arr_range = arr_max - arr_min
    scaled = np.array((arr-arr_min) / float(arr_range), dtype='f')
    arr_new = -1 + (scaled * 2)
    # Make sure min value is -1 and max value is 1
    #print('Min: %.3f, Max: %.3f' % (arr_new.min(), arr_new.max()))
    return arr_new


def _preprocess(img):

    # convert 16bit images to 8bit range (0-1)
    img = (img) * 255.0 / 65535

    # Wu et al. <- mean subtraction & divide by std
    img -= np.mean(img)
    img /= np.maximum(np.std(img), 10 ** (-5))

    img = _normalize(img)

    return img


import pandas as pd

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, preprocess, ignore_label=None):
        self.transform = transform
        self.preprocess = preprocess

        # kalau main_dir list → ambil item pertama
        if isinstance(main_dir, list):
            main_dir = main_dir[0]

        index_file = os.path.join(main_dir, "index.csv")
        if os.path.exists(index_file):
            # === gunakan index.csv (format: path,label,domain,idx) ===
            df = pd.read_csv(index_file)
            self.total_imgs = df['path'].tolist()
            self.labels = df['label'].astype(int).tolist()
            self.domains = df['domain'].tolist()
            self.indices = df['idx'].astype(int).tolist()
        else:
            # === fallback ke cara lama (folder normal/benign/malignant) ===
            self.total_imgs = get_filenames([main_dir], ignore_label)
            self.labels = _get_all_labels(self.total_imgs)
            self.domains = _get_all_domains(self.total_imgs)
            self.indices = list(range(len(self.total_imgs)))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_fname = self.total_imgs[idx]
        numpy_image = io.imread(img_fname)

        if self.preprocess:
            numpy_image = _preprocess(numpy_image)

        tensor_image = self.transform(numpy_image)

        # ambil label/domain dari index.csv kalau ada
        if hasattr(self, 'labels'):
            label = self.labels[idx]
        else:
            label = get_class(img_fname)

        if hasattr(self, 'domains'):
            domain = self.domains[idx]
        else:
            domain = get_domain(img_fname)

        return (tensor_image, label, domain, idx)


# load training and val datasets
def load_data(training_dirs, preprocess, ignore_label,
              data_seed, val_split=0.15):
    dataset = CustomDataSet(main_dir=training_dirs, preprocess=preprocess, transform=params.data_transform,
                            ignore_label=ignore_label)

    dataset_size = len(dataset.total_imgs)
    indeces = list(range(dataset_size))

    train_imgs, val_imgs, train_labels, val_labels, \
    train_domains, val_domains, train_idx, val_idx = train_test_split(dataset.total_imgs,
                                                                      dataset.labels,
                                                                      dataset.domains,
                                                                      indeces,
                                                                      random_state=data_seed,
                                                                      train_size=1-val_split,
                                                                      test_size=val_split,
                                                                      #stratify=dataset.labels)
                                                                      stratify=np.stack((dataset.labels, dataset.domains),).T)

    trainset = SubsetWithIndex(dataset, train_idx)
    valset = SubsetWithIndex(dataset, val_idx)

    return trainset, valset


class WeightedSubsetRandomSampler(Sampler):
    r"""Samples elements from a given list of indices with given probabilities (weights), with replacement.

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, indices, weights, num_samples=0):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool):
            raise ValueError("num_samples should be a non-negative integeral "
                             "value, but got num_samples={}".format(num_samples))
        self.indices = indices
        weights = [ weights[i] for i in self.indices ]
        self.weights = torch.tensor(weights, dtype=torch.double)
        if num_samples == 0:
            self.num_samples = len(self.weights)
        else:
            self.num_samples = num_samples
        self.replacement = True

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples



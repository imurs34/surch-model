import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pickle
import numpy as np

from .augmentations import *

def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class DataIterator(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels[:,0]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        #####
        f = int(int(os.path.basename(img_names).strip('.jpg')))
        img_names = os.path.join(os.path.dirname(img_names), str(f)+'.jpg')
        #print(img_names)
        #####
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase

    def __len__(self):
        return len(self.file_paths)


class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

def get_data(data_path, use_flip=1, crop_type=1, sequence_length=10, val_batch_size=320, train_batch_size=200,
             workers=32):

    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)

    train_transforms, test_transforms = None, None
    # Train transforms
    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])

    # Test transforms
    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])(crop) for crop in crops]))
        ])

    train_dataset = DataIterator(train_paths_80, train_labels_80, train_transforms)
    val_dataset = DataIterator(val_paths_80, val_labels_80, test_transforms)

    # ===

    train_useful_start_idx_80 = get_useful_start_idx(sequence_length, train_num_each_80)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each_80)

    train_we_use_start_idx_80 = train_useful_start_idx_80

    num_train_we_use_80 = len(train_useful_start_idx_80)
    num_val_we_use = len(val_useful_start_idx)

    train_idx = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx.append(train_useful_start_idx_80[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_useful_start_idx[i] + j)

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=False
    )
    train_idx_80 = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx_80.append(train_useful_start_idx_80[i] + j)

    def shuffle_train():
        np.random.shuffle(train_we_use_start_idx_80)
        train_idx_80 = []
        for i in range(num_train_we_use_80):
            for j in range(sequence_length):
                train_idx_80.append(train_we_use_start_idx_80[i] + j)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset, train_idx_80),
            num_workers=workers,
            pin_memory=False
        )
        return train_loader

    classes = {'BN': 0, 'BNU': 1, 'DVC': 2, 'LN': 3, 'NVB': 4, 'PA': 5, 'SVVD': 6, 'etc': 7}

    return shuffle_train, val_loader, classes

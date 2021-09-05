import os 
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch

def SaveResults(path, list):
    with open(path, 'w+') as f:
        for _ in list:
            print(_)
            f.writelines(str(_)+'\n')

class ImportDataset(Dataset):
    def __init__(self, img, label, transform=None, target_transform=None):
        self.img = img
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        if self.img[idx].shape[0] == 3:
            image = Image.fromarray(torch.tensor(self.img[idx]).squeeze().permute(1,2,0).numpy())       
        else:
            image = Image.fromarray(self.img[idx])
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class CustomImageDataset(ImportDataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.img = []
        self.label = []
        for _ in samples:
            path, target = _
            img_ = self.pil_loader(path)
            img_arry = np.array(img_)
            self.img.append(img_arry)
            self.label.append(int(target))

        self.img = np.array(self.img)
        self.label = np.array(self.label)
        self.label = self.label - min(self.label)
        self.transform = transform
        self.target_transform = target_transform

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

def plot_class(source_dataset, target_dataset):
    figure = plt.figure(figsize=(24, 8))
    cols, rows = 9, 3
    for i in range(1, cols * rows + 1):
        if i%9 in [1,2,3,4]:
            sample_idx = torch.randint(len(source_dataset), size=(1,)).item()
            img, label = source_dataset[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title('scr:'+str(label), color='C0', fontsize=16, fontweight='bold')
        else:
            sample_idx = torch.randint(len(target_dataset), size=(1,)).item()
            img, label = target_dataset[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title('tar:'+str(label), color='C2', fontsize=16, fontweight='bold')
        plt.axis("off")
        plt.imshow(img.squeeze().permute(1,2,0))
    plt.tight_layout()
    plt.show()



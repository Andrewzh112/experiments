import pandas as pd
import torch
from random import shuffle
from torch.utils.data import Dataset


class CombinedMNIST(Dataset):
    def __init__(self, data_path, num_classes=10):
        self.num_classes = num_classes
        self.dataset = pd.read_csv(data_path)

    def __len__(self):
        return self.dataset.label.value_counts().min() // 2 * self.num_classes

    def __getitem__(self, idx):
        label, i =  idx % self.num_classes, idx // self.num_classes * 2
        digit1 = self.dataset[self.dataset.label == label].iloc[i, 1:].values
        digit2 = self.dataset[self.dataset.label == label].iloc[i + 1, 1:].values
        return (
            torch.from_numpy(digit1).view(1, 28, 28).float(),
            torch.from_numpy(digit2).view(1, 28, 28).float(),
            torch.tensor([label]).float())

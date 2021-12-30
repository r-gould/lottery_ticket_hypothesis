import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MNIST(Dataset):
    def __init__(self, root, mode, transform):
        self.root = root
        self.mode = mode
        self.transform = transforms.Compose(transform)

        self.data_path = root + "/" + mode + ".csv"
        self.data = pd.read_csv(self.data_path).to_numpy()

    def __getitem__(self, index):
        label = self.data[index, 0].astype(np.longlong)
        image = self.data[index, 1:].reshape(28, 28, -1) / 255
        image = self.transform(image)
        return (image, label)

    def __len__(self):
        return self.data.shape[0]
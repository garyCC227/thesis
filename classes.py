from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os 
import json
import pandas as pd
from pathlib import Path

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = imgs
        self.labels = labels
#         self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.abspath(self.imgs[idx])
        image = io.imread(img_name)
        landmarks = self.labels[idx]
        sample = {'image': image, 'landmarks':landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
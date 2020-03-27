import logging
import math
import numpy as np
import os
import random
import tensorflow as tf

from data.tiny_imagenet.read_tiny_imagenet import TinyImageNet


import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy

import json
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


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
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if new_h > h:
            top = np.random.randint(0, new_h - h)
        elif new_h == h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)

        if new_w > w:
            left = np.random.randint(0, new_w - w)
        elif new_w == w:
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': landmarks}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    num_data = 50
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for index, batch in enumerate(dataloaders[phase]):
                if index > num_data:
                    break

                inputs = batch['image']
                labels = batch['landmarks']
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.int64)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            epoch_loss = running_loss / num_data
            epoch_acc = running_corrects.double() / num_data
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def format_example(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image


if __name__ == "__main__":

    torch.cuda.empty_cache()
    #path ="E:/code/adapted_deep_embeddings/data/tiny_imagenet/tiny-imagenet-200"
#   path = "/srv/scratch/z5163479/tiny-imagenet-200"
#   dataset = TinyImageNet(path)
    # train_path = "E:\\aptos\\labelsbase15.json"
    # test_path = "E:\\aptos\\labelsval15.json"
    train_path = "/home/z5163479/code/base15.json"
    test_path = "/home/z5163479/code/val15.json"

    with open(train_path) as json_data:
        train_data = json.load(json_data)

    with open(test_path) as json_data:
        test_data = json.load(json_data)

    train_labels = train_data['image_labels']
    train_imgs = train_data['image_names']

    test_labels = test_data['image_labels']
    test_imgs = test_data['image_names']

    train_dataset = FaceLandmarksDataset(train_imgs, train_labels,
                                            transform=transforms.Compose([
                                                Rescale(224),
                                                RandomCrop(224),
                                                ToTensor()
                                                ]))
    test_dataset = FaceLandmarksDataset(test_imgs, test_labels,
                                            transform=transforms.Compose([
                                                Rescale(224),
                                                RandomCrop(224),
                                                ToTensor()
                                                ]))    

    dataset_sizes = {'train':len(train_dataset), 'val':len(test_dataset)}

    train_dataloader = torch.utils.data.DataLoader(train_dataset , batch_size=32, num_workers=0, shuffle=True) 
    test_dataloader = torch.utils.data.DataLoader(test_dataset , batch_size=32, num_workers=0, shuffle=True) 
    
    dataloaders = {'train':train_dataloader, 'val':test_dataloader}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 5)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=10)
    # for i in test_dataloader:
    #     print(i['image'].shape, i['landmarks'].shape)
    #     break

    # for i in range(len(train_dataset)):
    #     sample = train_dataset[i]

    #     print(i, sample['image'].size(), sample["landmarks"])

    #     if i == 3:
    #         break


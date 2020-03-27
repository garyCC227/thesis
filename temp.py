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
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
  path = "/srv/scratch/z5163479/tiny-imagenet-200"
  dataset = TinyImageNet(path)

  (x_train, y_train), (x_test, y_test) = dataset.kntl_data_form(350, 5, 10, 5)

  np.random.seed(1)

  shuffle = np.random.permutation(len(y_train))
  batch_size = 32
  learning_rate = 0.005

  IMG_SIZE = 224 # All images will be resized to 160x160
  print("shape before:{}".format(x_train.shape))
  x_train=format_example(x_train)
  # x_valid=format_example(x_valid)
  x_test=format_example(x_test)
  print(x_train.shape)


  x_train = tf.Session().run(x_train)
  # y_train = tf.Session().run(y_train)
  x_test = tf.Session().run(x_test)
  # y_test = tf.Session().run(y_test)

  x_train = x_train.transpose(0,3,1,2)
  x_test = x_test.transpose(0,3,1,2)

  x_train = torch.Tensor(x_train) # transform to torch tensor
  y_train = torch.Tensor(y_train)
  x_test = torch.Tensor(x_test) # transform to torch tensor
  y_test = torch.Tensor(y_test)


  train_data = data.TensorDataset(x_train,y_train) # create your datset
  test_data = data.TensorDataset(x_test,y_test) # create your datset
  dataset_sizes = {'train':len(train_data), 'val':len(test_data)}

  train_dataloader = data.DataLoader(train_data, batch_size=32, num_workers=4, shuffle=True) 
  test_dataloader = data.DataLoader(test_data, batch_size=32, num_workers=4, shuffle=True) 

  dataloaders = {'train':train_dataloader, 'val':test_dataloader}
  


  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_ft = models.resnet50(pretrained=False)
  num_ftrs = model_ft.fc.in_features
  # Here the size of each output sample is set to 2.
  # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
  model_ft.fc = nn.Linear(num_ftrs, 200)

  model_ft = model_ft.to(device)

  criterion = nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

  model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)
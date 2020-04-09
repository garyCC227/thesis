import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
from PIL import *
import matplotlib.image as mpimg
import numpy as np
from keras.preprocessing import image
import json
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, ReLU, MaxPool2D,InputLayer
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU   
from keras import optimizers, regularizers
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard
import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.applications import DenseNet121
from keras import layers
import sys



np.random.seed(2019)
tf.set_random_seed(2019)

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im

def create_datagen():
  return ImageDataGenerator(
      zoom_range=0.15,  # set range for random zoom
      # set mode for filling points outside the input boundaries
      fill_mode='constant',
      cval=0.,  # value used for fill_mode = "constant"
      horizontal_flip=True,  # randomly flip images
      vertical_flip=True,  # randomly flip images
  )


'''
resnet50
'''
def build_model(input_shape):
  
  base_model =ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
  #for layer in  base_model.layers[:10]:
    #layer.trainable = False
    #layer.padding='same'
 
  #for layer in  base_model.layers[10:]:
    #layer.trainable = True
    #layer.padding='same'
    
#   x = base_model.get_layer('avg_pool').output
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  # x = BatchNormalization()(x)
  x = Dropout(0.5)(x)

#   x = Flatten() (x)
#   x = Dropout(0.5)(x)
  # x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
#   x = BatchNormalization()(x)
#   x = Dropout(0.5)(x)
#   x = Dense(32, activation='relu')(x)
  # x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  # x = Dropout(0.5)(x)
  # x = BatchNormalization()(x)
  # x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  # x = Dropout(0.5)(x)
  # x = BatchNormalization()(x)
#   x = Dense(512, activation='relu')(x)
  # x = LeakyReLU(alpha=0.1)(x)
    
#   x = Dropout(0.3)(x)
  #x = Dense(5, activation='softmax')(x)
  #model = Model(base_model.input, x)
  predictions = Dense(5, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
#   for layer in model.layers[:-2]:
#     layer.trainable = False

  model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
  )
  return model



# def build_model(input_shape):
#     densenet = DenseNet121(
#       weights='/home/z5163479/code/adapted_deep_embeddings/DenseNet-BC-121-32-no-top.h5',
#       include_top=False,
#       input_shape=input_shape
#     )
#     model = Sequential()
#     model.add(densenet)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(5, activation='sigmoid'))
    
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(lr=0.00005),
#         metrics=['accuracy']
#     )
    
#     return model

def get_preds(arr):
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)


def main():
  train_df = pd.read_csv('/srv/scratch/z5163479/aptos/labels/trainLabels19.csv')
  print(train_df.shape)
  # train_df.head()
  N = train_df.shape[0]
  x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

  for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        f'/srv/scratch/z5163479/aptos/resized_train_19/{image_id}.jpg'
    )
  
  y_train = pd.get_dummies(train_df['diagnosis']).values

  print(x_train.shape)
  print(y_train.shape)

  x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.15, 
    random_state=2019
  )

  y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
  y_train_multi[:, 4] = y_train[:, 4]

  for i in range(3, -1, -1):
      y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

  print("Original y_train:", y_train.sum(axis=0))
  print("Multilabel version:", y_train_multi.sum(axis=0))

  assert (np.argmax(y_train, 1) == get_preds(y_train_multi)).all()

  y_val_multi = np.empty(y_val.shape, dtype=y_val.dtype)
  y_val_multi[:, 4] = y_val[:, 4]

  for i in range(3, -1, -1):
      y_val_multi[:, i] = np.logical_or(y_val[:, i], y_val_multi[:, i+1])

  print("Original y_val:", y_train.sum(axis=0))
  print("Multilabel version:", y_train_multi.sum(axis=0))

  assert (np.argmax(y_val, 1) == get_preds(y_val_multi)).all()

  BATCH_SIZE = 32

  # Using original generator
  data_generator = create_datagen().flow(x_train, y_train_multi, batch_size=BATCH_SIZE, seed=2019)
  # Using Mixup
  model = build_model((224,224,3))
  model.summary()

  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'aptos2019'
  tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                        write_graph=True, write_images=False)
  # mixup_generator = MixupGenerator(x_train, y_train, batch_size=BATCH_SIZE, alpha=0.2, datagen=create_datagen())()
  history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=20,
    validation_data=(x_val, y_val_multi),
    callbacks=[tensorboard]
  )


  (loss, accuracy) = model.evaluate(x_val, y_val_multi, batch_size=64, verbose=1)
  print( 'loss = {:.4f}, accuracy: {:.4f}%'.format(loss,accuracy*100))
  test_pred = model.predict(x_val, verbose=1)
  test_true=y_val.argmax(axis=1) 
# print(classification_report(test_true, test_pred, target_names=["0","1","2","3","4"]))

  arr = 1 * (test_pred > 0.5)
  test_pred = get_preds(arr)
  # test_true = get_preds(y_val)
  print(classification_report(test_true, test_pred, target_names=["0","1","2","3","4"]))



  print("For train")
  (loss, accuracy) = model.evaluate(x_train, y_train_multi, batch_size=64, verbose=1)
  print( 'loss = {:.4f}, accuracy: {:.4f}%'.format(loss,accuracy*100))
  test_pred = model.predict(x_train, verbose=1)
  test_true=y_train.argmax(axis=1) 
# print(classification_report(test_true, test_pred, target_names=["0","1","2","3","4"]))

  arr = 1 * (test_pred > 0.5)
  test_pred = get_preds(arr)
  # test_true = get_preds(y_val)
  print(classification_report(test_true, test_pred, target_names=["0","1","2","3","4"]))

if __name__ == "__main__":
    main()
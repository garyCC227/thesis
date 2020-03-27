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
from keras.applications.resnet import ResNet152
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
import imgaug.augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16


import sys
# sys.stdout = open('./code/adapted_deep_embeddings/Wholedata_log.txt','wt')
# def get_model(input_shape):
#   kernel_size = 7
#   model = Sequential([
#     InputLayer(input_shape=input_shape),
#     Conv2D(32,kernel_size ),
#     BatchNormalization(),
#     ReLU(),
#     MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#     Conv2D(64,kernel_size , input_shape=input_shape),
#     BatchNormalization(),
#     ReLU(),
#     MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#      Conv2D(128,kernel_size , input_shape=input_shape),
#     BatchNormalization(),
#     ReLU(),
#     MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#     Conv2D(512,kernel_size , input_shape=input_shape),
#     BatchNormalization(),
#     ReLU(),
#     GlobalAveragePooling2D(),
#     Dense(5, activation='softmax'),
#   ])
#   return model

def get_model(input_shape):
  
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
#   x = Dropout(0.5)(x)

#   x = Flatten() (x)
#   x = Dropout(0.5)(x)
  x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
#   # x = BatchNormalization()(x)
#   x = Dropout(0.5)(x)
#   x = Dense(32, activation='relu')(x)
  # x = Dense(128, activation='relu')(x)
  # x = Dropout(0.5)(x)
#   x = Dense(2048, activation='relu')(x)
#   x = Dense(512, activation='relu')(x)
#   x = LeakyReLU(alpha=0.1)(x)
    
  x = Dropout(0.5)(x)
  #x = Dense(5, activation='softmax')(x)
  #model = Model(base_model.input, x)
  predictions = Dense(5, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  for layer in model.layers[:-4]:
    layer.trainable = False

  return model

def split_data(data):
    random.shuffle(data)
    img_train, testset = train_test_split(data, test_size=0.2)
    trainset, valset= train_test_split(img_train,test_size=0.1)
    
    return trainset, valset, testset

def create_data(images_path, labels):
    data = []
    zip_data = list(zip(labels, images_path))
    random.shuffle(zip_data)
    zip_data = zip_data[:7000]

    for label, img_path in zip_data:
        # print(img_paths)
        img = image.load_img(img_path, target_size=(224,224))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        data.append([img, label])

    return data

def check_num_each_class(train_y, test_y, val_y):
    zero, one, two, three, four = 0, 0, 0, 0, 0
    for e in train_y:
        if e == 0:
            zero += 1
        elif e == 1:
            one += 1
        elif e == 2:
            two += 1
        elif e == 3:
            three += 1
        elif e == 4:
            four += 1

    print("each classes has # images in train:\n")
    print(zero, one, two, three, four)

    zero, one, two, three, four = 0, 0, 0, 0, 0
    for e in test_y:
        if e == 0:
            zero += 1
        elif e == 1:
            one += 1
        elif e == 2:
            two += 1
        elif e == 3:
            three += 1
        elif e == 4:
            four += 1

    print("each classes has # images in train:\n")
    print(zero, one, two, three, four)

    zero, one, two, three, four = 0, 0, 0, 0, 0
    for e in val_y:
        if e == 0:
            zero += 1
        elif e == 1:
            one += 1
        elif e == 2:
            two += 1
        elif e == 3:
            three += 1
        elif e == 4:
            four += 1

    print("each classes has # images in train:\n")
    print(zero, one, two, three, four)

def main():
    # path = 'E:\\aptos\\labelsbase15.json'
    print("CBR-------------Whole dataset")
    classes = 5  #TODO:
    BS = 32 #batch size
    path_base = "/home/z5163479/code/base15.json"
    path_novel = "/home/z5163479/code/novel15.json"
    path_val = "/home/z5163479/code/val15.json"
    with open(path_base, 'r') as f:
        data1 = json.load(f)
    with open(path_novel, 'r') as f:
        data2 = json.load(f)
    with open(path_val, 'r') as f:
        data3 = json.load(f)


    labels = data1['image_labels']+ data2['image_labels'] + data3['image_labels']
    images_path = data1['image_names'] + data2['image_names'] + data3['image_names']

    # assert len(labels) == 35126
    assert len(images_path) == len(labels)
    print("Num of images: {}\n".format(len(labels)))

    epoch = 125
    NN_layer = "includeT_{}classes_binaryCross_sigmoid_wholedataset_epoch{}_imagenet".format(classes,epoch) #TODO
    print(NN_layer)
    
    data = create_data(images_path, labels)
    img_train, val_test, img_test = split_data(data)

    print(len(val_test))
    print(len(img_train))
    print(len(img_test))
    val_size = len(val_test)
    train_size = len(img_train)
    test_size = len(img_test)
    # assert val_size + train_size + test_size == 35126
    

    val_x = []
    val_y = []
    for features, label in val_test:
        val_x.append(features)
        val_y.append(label)
        
    val_x=np.array(val_x).reshape(val_size,224,224,3)
    # val_x = val_x.astype('float32') / 255

    train_x = []
    train_y = []
    for features, label in img_train:
        train_x.append(features)
        train_y.append(label)
        
    train_x=np.array(train_x).reshape(train_size,224,224,3)
    # train_x = train_x.astype('float32') / 255
#
    test_x = []
    test_y = []
    for features, label in img_test:
        test_x.append(features)
        test_y.append(label)
        
    test_x=np.array(test_x).reshape(test_size,224,224,3)
    # test_x = test_x.astype('float32')/255

    check_num_each_class(train_y, test_y, val_y)

    # train_y=to_categorical(train_y)
    # test_y=to_categorical(test_y)
    # val_y=to_categorical(val_y)

    model50 = get_model(input_shape=(224,224,3))
    model50.summary()
    adam = optimizers.Adam(lr=0.001)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)
    model50.compile(optimizer=adam,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    

    image_gen = ImageDataGenerator(
                                # horizontal_flip=True,
                                # vertical_flip=True,
                               data_format='channels_last')
    image_gen.fit(train_x)

    img_gen=image_gen.flow(train_x, train_y, batch_size=BS, shuffle=True)
    

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)
    train_model = model50.fit_generator(img_gen, validation_data=(val_x, val_y), epochs=epoch, steps_per_epoch=len(train_x)//BS, verbose=1, callbacks=[tensorboard])

    # train_model=model50.fit(train_x, train_y, batch_size=8,epochs=epoch,verbose=1,validation_data=(val_x, val_y),  callbacks=[tensorboard])
    
    (loss, accuracy) =  model50.evaluate(test_x, test_y, batch_size=10, verbose=1)
    print( 'loss = {:.4f}, accuracy: {:.4f}%'.format(loss,accuracy*100))


    test_pred = model50.predict(test_x, verbose=1, batch_size=64).argmax(axis=1)
    test_true=test_y
    print(test_pred)
    # print(train_model.Hisory.keys())
    print(classification_report(test_true, test_pred, target_names=["0","1","2","3","4"]))

    # plt.figure()
    # plt.plot(train_model.history['accuracy'])
    # plt.plot(train_model.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.savefig("./code/adapted_deep_embeddings/acc_{}-{}.png".format(NN_layer,k))
    # # plt.show()

    # plt.figure()
    # plt.plot(train_model.history['loss'])
    # plt.plot(train_model.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.savefig("./code/adapted_deep_embeddings/loss_{}-{}.png".format(NN_layer,k))
    # # plt.show()

if __name__ == "__main__":
    main()

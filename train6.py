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

import sys
# sys.stdout = open('./code/adapted_deep_embeddings/log.txt','wt')

# def check_num_each_class(train_y, test_y, val_y):
#     zero, one, two, three, four = 0, 0, 0, 0, 0
#     for e in train_y:
#         if e == 0:
#             zero += 1
#         elif e == 1:
#             one += 1
#         elif e == 2:
#             two += 1
#         elif e == 3:
#             three += 1
#         elif e == 4:
#             four += 1

#     print("each classes has # images in train:\n")
#     print(zero, one, two, three, four)

#     zero, one, two, three, four = 0, 0, 0, 0, 0
#     for e in test_y:
#         if e == 0:
#             zero += 1
#         elif e == 1:
#             one += 1
#         elif e == 2:
#             two += 1
#         elif e == 3:
#             three += 1
#         elif e == 4:
#             four += 1

#     print("each classes has # images in train:\n")
#     print(zero, one, two, three, four)

#     zero, one, two, three, four = 0, 0, 0, 0, 0
#     for e in val_y:
#         if e == 0:
#             zero += 1
#         elif e == 1:
#             one += 1
#         elif e == 2:
#             two += 1
#         elif e == 3:
#             three += 1
#         elif e == 4:
#             four += 1

#     print("each classes has # images in train:\n")
#     print(zero, one, two, three, four)

# def get_model(input_shape):
#   kernel_size = 5
#   model = Sequential([
#     InputLayer(input_shape=input_shape),
#     Conv2D(64,kernel_size ),
#     BatchNormalization(),
#     ReLU(),
#     MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#     Conv2D(128,kernel_size , input_shape=input_shape),
#     BatchNormalization(),
#     ReLU(),
#     MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#     Conv2D(256,kernel_size , input_shape=input_shape),
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

'''
resnet50
'''
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
  # x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
#   x = BatchNormalization()(x)
#   x = Dropout(0.5)(x)
#   x = Dense(32, activation='relu')(x)
  x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)
  x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)
#   x = Dense(512, activation='relu')(x)
  # x = LeakyReLU(alpha=0.1)(x)
    
#   x = Dropout(0.3)(x)
  #x = Dense(5, activation='softmax')(x)
  #model = Model(base_model.input, x)
  predictions = Dense(5, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
#   for layer in model.layers[:-2]:
#     layer.trainable = False

  return model

def split_data(data_dict):
    trainset = []
    valset = []
    testset=[]
    for label, images in data_dict.items():
        random.shuffle(images)
        img_train, img_test = train_test_split(images, test_size=0.2)
        img_train, img_val = train_test_split(img_train,test_size=0.1)
        trainset = trainset + img_train
        valset = valset + img_val
        testset = testset + img_test
    
    return trainset, valset, testset
def create_data(images_dict):
    data = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[]
    }
    for label, img_paths in images_dict.items():
        for img_path in img_paths:
            img = image.load_img(img_path, target_size=(224,224))
            img = image.img_to_array(img)
            # img = preprocess_input(img)
            data[label].append([img, label])

    return data

def create_custom_gen(img_gen):
    seq = iaa.Sequential([
        iaa.MultiplyHue((0.5, 1.5)),
        iaa.imgcorruptlike.Contrast(severity=1)
    ])
    for X_batch, y_batch in img_gen:
        hue = seq(images = X_batch.astype(np.uint8))
        yield hue, y_batch
        # yield hue.astype('float32')/255.0, y_batch

def main():
    # path = 'E:\\aptos\\labelsbase15.json'
    classes = 5  #TODO:
    path_base = "/home/z5163479/code/base15.json"
    path_novel = "/home/z5163479/code/novel15.json"
    with open(path_base, 'r') as f:
        data = json.load(f)

    labels = np.array(data['image_labels'])
    images = np.array(data['image_names'])
    
    k=500#TODO
    epoch = 180
    NN_layer = "resnet_{}classes_SparseCross_sigmoid_{}_epoch{}_imagenet".format(classes,k,epoch) #TODO
    BS = 32 #batch size
    print(NN_layer)
    # NN_layer = "Flaten"
    # print("drop out with global pool {}\n".format(k))
    # print("normal flaten {}\n".format(k))
    # random.shuffle(images) #TODO:
    zero_images = images[labels == 0][:k]
    one_images = images[labels == 1][:k]
    two_images = images[labels == 2][:k]
    three_images = images[labels == 3][:k]
    four_images1 = images[labels == 4][:k]

    # add more 
    four_images2 = []

    if len(four_images1) < k :
        # path = 'E:\\aptos\\labelsnovel15.json'
        print("adding image from second dataset\n")
        path_novel = "/home/z5163479/code/novel15.json"
        with open(path_base, 'r') as f:
            add_data = json.load(f)
        add_labels = np.array(add_data['image_labels'])
        add_images = np.array(add_data['image_names'])
        n = k - len(four_images1)
        four_images2 = add_images[add_labels == 4][:n]

    four_images = [y for x in [four_images1, four_images2] for y in x]
    print("0 images: {}, four images: {}".format(len(zero_images), len(four_images)))

    # print(zero_images.shape)
    images_dict = {
        0:zero_images,
        1:one_images,
        2:two_images,
        3:three_images,
        4:four_images
    }

    data = create_data(images_dict)
    img_train, val_test, img_test = split_data(data)

    print(len(val_test))
    print(len(img_train))
    print(len(img_test))
    val_size = len(val_test)
    train_size = len(img_train)
    test_size = len(img_test)
    # assert val_size + train_size + test_size == k * classes
     
    val_x = []
    val_y = []
    random.shuffle(val_test)
    for features, label in val_test:
        features = preprocess_input(features)
        val_x.append(features)
        val_y.append(label)
        
    val_x=np.array(val_x).reshape(val_size,224,224,3)
    # val_x = val_x.astype('float32') / 255.0

    train_x = []
    train_y = []
    random.shuffle(img_train)
    for features, label in img_train:
        features = preprocess_input(features)
        train_x.append(features)
        train_y.append(label)
        
    train_x=np.array(train_x).reshape(train_size,224,224,3)
    # train_x = train_x.astype('float32') / 255.0

    test_x = []
    test_y = []
    random.shuffle(img_test)
    for features, label in img_test:
        features = preprocess_input(features)
        test_x.append(features)
        test_y.append(label)
        
    test_x=np.array(test_x).reshape(test_size,224,224,3)
    # test_x = test_x.astype('float32')/255.0

    train_y=to_categorical(train_y)
    test_y=to_categorical(test_y)
    val_y=to_categorical(val_y)

    image_gen = ImageDataGenerator(
#                               rescale=1./255
                            # rotation_range=45,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
#                             zoom_range=0.2,
#                             shear_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=True,
#                             fill_mode='nearest'
                           data_format='channels_last'
                           )
 
    img_gen=image_gen.flow(train_x, train_y, batch_size=BS, shuffle=True)
    # img_gen = create_custom_gen(img_gen)
    test_datagen = ImageDataGenerator()

    val_gen=test_datagen.flow(val_x, val_y, batch_size=32, shuffle=False)
    # test_gen=test_datagen.flow(test_x, test_y, batch_size=20, shuffle=False)

    # check_num_each_class(train_y, test_y, val_y)

    # for val_batch, Y in val_gen:
    #   for i in range(10):
    #     print(val_batch[i], Y[i])
    #     break
    #   break
    # return
    
    # count = 0
    # for X_batch, y in img_gen:
    #     zero = 0; two = 0; three = 0; one = 0; four =0;
    #     for i in range(BS):
    #         if y[i] == 0:
    #             zero += 1
    #         elif y[i] == 1:
    #             one += 1
    #         elif y[i] == 2:
    #             two += 2
    #         elif y[i] == 3:
    #             three += 3
    #         elif y[i] == 4:
    #             four += 4
        
    #     print("class:\n")
    #     print(zero, one, two, three, four )
    #     if count == 5:
    #         break
    #     count += 1
    # return

    model50 = get_model(input_shape=(224,224,3))
    model50.summary()
    adam = optimizers.Adam(lr=0.001)
    model50.compile(
                    # optimizer=adam,
                  optimizers.RMSprop(lr=2e-5),
#                 optimizer=sgd2,
                    # loss='categorical_crossentropy',
                    # loss='kullback_leibler_divergence',
                    loss= 'binary_crossentropy',
                        # loss='sparse_categorical_crossentropy',
                        metrics=['acc'])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)
    

    # train_model=model50.fit(train_x, train_y, batch_size=8,epochs=epoch,verbose=1,validation_data=(val_x, val_y), callbacks=[tensorboard])
    # train_model = model50.fit_generator(img_gen, validation_data=val_gen,validation_steps = len(val_x)//32, epochs=10, steps_per_epoch=len(train_x)//BS, verbose=1)

    # ## start train
    # for layer in model50.layers[:165]:
    #   layer.trainable = False
    # for layer in model50.layers[165:]:
    #   layer.trainable = True

    # model50.compile(optimizer=optimizers.Adam(lr=0.01)  ,
    #                     loss='sparse_categorical_crossentropy',
    #                     metrics=['accuracy'])
    
    train_model = model50.fit_generator(img_gen, validation_data=(val_gen),validation_steps = len(val_x)//32, epochs=epoch, steps_per_epoch=len(train_x)//BS, verbose=1, callbacks=[tensorboard])

    (loss, accuracy) = model50.evaluate(test_x, test_y, batch_size=64, verbose=1)
    print( 'loss = {:.4f}, accuracy: {:.4f}%'.format(loss,accuracy*100))


    test_pred = model50.predict(test_x, verbose=1, batch_size=64).argmax(axis=1)
    test_true=test_y.argmax(axis=1) 
    # test_true=test_y
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

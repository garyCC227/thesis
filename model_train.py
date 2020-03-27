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
# from keras.applications.vgg16 import preprocess_input

import sys
# sys.stdout = open('./code/adapted_deep_embeddings/log.txt','wt')




'''
Get Tiny ImageNet dataset
'''
# class TinyImageNet():
#     def __init__(self, path):
#         self.path = path

#         self.images = []
#         self.labels = []

#     def load_data(self):
#         ims, labels = self.load(self.path)

#         self.images = self.process_images(ims)
#         self.labels = self.process_labels(labels)

#         return self.images, self.labels

#     def process_images(self, images):
#         images_np = np.array(images) / 255.0
#         return images_np

#     def process_labels(self, labels):
#         return np.array(labels)

#     @classmethod
#     def load(cls, path):
#         class_id = 0
#         id_to_label = {}
#         validation_annotations = None
#         validation_images = {}
#         images = []
#         labels = []
#         for root, dirs, files in os.walk(path):
#             for f in files:
#                 if f == 'val_annotations.txt':
#                     validation_annotations = os.path.join(root, f)
#                 elif f.endswith('.JPEG'):
#                     path = os.path.join(root, f)
#                     id = f.split('_')[0]
#                     if id == 'val':
#                         validation_images[f] = path
#                     else:
#                         if id not in id_to_label:
#                             id_to_label[id] = class_id
#                             class_id += 1
#                         img = image.load_img(path, target_size=(64,64))
#                         img = image.img_to_array(img)
#                         if len(img.shape) == 2:
#                             img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
#                         images.append(img)
#                         labels.append(id_to_label[id])

#         with open(validation_annotations) as val_ann:
#             for line in val_ann:
#                 contents = line.split()
#                 img = image.load_img(validation_images[contents[0]], target_size=(64,64))
#                 img = image.img_to_array(img)
#                 if len(img.shape) == 2:
#                     img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
#                 images.append(img)
#                 labels.append(id_to_label[contents[1]])

#         return images, labels

#     def kntl_data_form(self, k1, n1, k2, n2):
#         assert n1 + n2 <= 200
#         assert k1 < 550 and k2 < 550
#         self.load_data()

#         print('Full dataset: {0}'.format(len(self.labels)))

#         all_classes = np.unique(self.labels)
#         print('Number of classes: {0}'.format(len(all_classes)))

#         task2_classes = np.sort(np.random.choice(all_classes, n2, replace=False))
#         all_classes = np.delete(all_classes, np.where(np.isin(all_classes, task2_classes)))
#         indices = np.isin(self.labels, task2_classes)
#         self.x_task2, self.y_task2 = self.images[indices], self.labels[indices]
#         shuffle = np.random.permutation(len(self.y_task2))
#         self.x_task2, self.y_task2 = self.x_task2[shuffle], self.y_task2[shuffle]

#         task1_classes = np.sort(np.random.choice(all_classes, n1, replace=False))
#         indices = np.isin(self.labels, task1_classes)
#         self.x_task1, self.y_task1 = self.images[indices], self.labels[indices]
#         shuffle = np.random.permutation(len(self.y_task1))
#         self.x_task1, self.y_task1 = self.x_task1[shuffle], self.y_task1[shuffle]

#         # print('Task 1 Full: {0}'.format(len(self.y_task1)))
#         # print('Task 2 Full: {0}\n'.format(len(self.y_task2)))

#         # Force class labels to start from 0 and increment upwards by 1
#         sorted_class_indices = np.sort(np.unique(self.y_task1))
#         zero_based_classes = np.arange(0, len(sorted_class_indices))
#         for i in range(len(self.y_task1)):
#             self.y_task1[i] = zero_based_classes[sorted_class_indices == self.y_task1[i]]

#         self.x_train_task1 = []
#         self.y_train_task1 = []
#         self.x_valid_task1 = []
#         self.y_valid_task1 = []

#         for i in zero_based_classes:
#             all_indices = np.where(self.y_task1 == i)[0]
#             idx = np.random.choice(all_indices, k1, replace=False)
#             self.x_train_task1.extend(self.x_task1[idx])
#             self.y_train_task1.extend(self.y_task1[idx])
#             all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
#             self.x_valid_task1.extend(self.x_task1[all_indices])
#             self.y_valid_task1.extend(self.y_task1[all_indices])

#         self.x_train_task1 = np.array(self.x_train_task1)
#         self.y_train_task1 = np.array(self.y_train_task1)
#         self.x_valid_task1 = np.array(self.x_valid_task1)
#         self.y_valid_task1 = np.array(self.y_valid_task1)

#         # print('Task 1 training: {0}'.format(len(self.x_train_task1)))
#         print('Task 1 validation: {0}'.format(len(self.x_valid_task1)))

#         # Force class labels to start from 0 and increment upwards by 1
#         sorted_class_indices = np.sort(np.unique(self.y_task2))
#         zero_based_classes = np.arange(0, len(sorted_class_indices))
#         for i in range(len(self.y_task2)):
#             self.y_task2[i] = zero_based_classes[sorted_class_indices == self.y_task2[i]]

#         self.x_train_task2 = []
#         self.y_train_task2 = []
#         for i in zero_based_classes:
#             idx = np.random.choice(np.where(self.y_task2 == i)[0], k2, replace=False)
#             self.x_train_task2.extend(self.x_task2[idx])
#             self.y_train_task2.extend(self.y_task2[idx])
#             self.x_task2 = np.delete(self.x_task2, idx, axis=0)
#             self.y_task2 = np.delete(self.y_task2, idx, axis=0)

#         self.x_train_task2 = np.array(self.x_train_task2)
#         self.y_train_task2 = np.array(self.y_train_task2)

#         k_test = 550 - k2

#         self.x_test_task2 = []
#         self.y_test_task2 = []
#         for i in zero_based_classes:
#             idx = np.random.choice(np.where(self.y_task2 == i)[0], k_test, replace=False)
#             self.x_test_task2.extend(self.x_task2[idx])
#             self.y_test_task2.extend(self.y_task2[idx])

#         self.x_test_task2 = np.array(self.x_test_task2)
#         self.y_test_task2 = np.array(self.y_test_task2)

#         # print('k = {0}, n = {1}'.format(k2, n2))
#         # print('Task 2 training: {0}'.format(len(self.x_train_task2)))
#         # print('Task 2 test: {0}\n'.format(len(self.x_test_task2)))

#         # return (self.x_train_task1, self.y_train_task1), (self.x_valid_task1, self.y_valid_task1), (self.x_train_task2, self.y_train_task2), (self.x_test_task2, self.y_test_task2)
#         return (self.x_train_task1, self.y_train_task1), (self.x_valid_task1, self.y_valid_task1)


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

    print("each classes has # images in test:\n")
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

    print("each classes has # images in val:\n")
    print(zero, one, two, three, four)

'''
CBR model
'''
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
#      Conv2D(256,kernel_size , input_shape=input_shape),
#     BatchNormalization(),
#     ReLU(),
#     MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#      Conv2D(512,kernel_size , input_shape=input_shape),
#     BatchNormalization(),
#     ReLU(),
#     GlobalAveragePooling2D(),
#     Dense(5, activation='sigmoid'),
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
#   # x = BatchNormalization()(x)
#   x = Dropout(0.5)(x)
#   x = Dense(32, activation='relu')(x)
  # x = Dense(128, activation='relu')(x)
  # x = Dropout(0.5)(x)
#   x = Dense(2048, activation='relu')(x)
#   x = Dense(512, activation='relu')(x)
  # x = LeakyReLU(alpha=0.1)(x)
    
  x = Dropout(0.5)(x)
  #x = Dense(5, activation='softmax')(x)
  #model = Model(base_model.input, x)
  predictions = Dense(5, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
#   for layer in model.layers[:-2]:
#     layer.trainable = False

  return model

'''
VGG16
'''
# def get_model(input_shape):
  
#   image_input = Input(shape = (224,224, 3))
#   model = VGG16(input_tensor = image_input, weights = 'imagenet')
#   # model = ResNet50(weights='imagenet', include_top=False, input_tensor = image_input)
#   # model.summary()
#   last_layer = model.get_layer('block5_pool').output
#   x = Flatten(name='flatten')(last_layer)
#   x = Dense(128, activation='relu', name='fc1')(x)
#   x = Dense(128, activation='relu', name='fc2')(x)
#   out = Dense(5, activation='softmax', name='output')(x)
#   custom_vgg_model2 = Model(image_input, out)
#   for layer in custom_vgg_model2.layers[:-3]:
#     layer.trainable = False

#   return custom_vgg_model2

def split_data(data_dict):
    trainset = []
    valset = []
    testset=[]
    for label, images in data_dict.items():
        random.shuffle(images) #shuffle each class
        img_train, img_test = train_test_split(images, test_size=0.2)
        img_train, img_val = train_test_split(img_train,test_size=0.2)
        trainset = trainset + img_train
        valset = valset + img_val
        testset = testset + img_test
    
    #three dataset are stored in order data[class0, class1 ... class4]
    #will do futher shuffle 
    return trainset, valset, testset

'''
output dataset like
data = {
        0:[[class_0_nparray, label_0]],
        1:[[class_1_nparray, label_1]],
        2:[[class_2_nparray, label_2]],
        3:[[class_3_nparray, label_3]],
        4:[[class_4_nparray, label_4]]
    }
'''
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
            img = image.load_img(img_path, target_size=(587,587))
            img = image.img_to_array(img)
            img = preprocess_input(img)
            data[label].append([img, label])

    return data

'''
custom data augmentation
'''
def create_custom_gen(img_gen):
    seq = iaa.Sequential([
        iaa.MultiplyHue((0.5, 1.5)),
        iaa.imgcorruptlike.Contrast(severity=1)
    ])
    for X_batch, y_batch in img_gen:
        hue = seq(images = X_batch.astype(np.uint8))
        yield hue, y_batch

def main():
    # path = 'E:\\aptos\\labelsbase15.json'
    classes = n = 5  #TODO:
    
    k= 500 #TODO
    path_base = "/home/z5163479/code/base15.json"
    path_novel = "/home/z5163479/code/novel15.json"
    with open(path_base, 'r') as f:
        data = json.load(f)

    labels = np.array(data['image_labels'])
    images = np.array(data['image_names'])
    
    epoch = 60
    NN_layer = "resnet_{}classes_SparseCross_sigmoid_{}_epoch{}_imagenet".format(classes,k,epoch) #TODO
    BS = 32 #batch size
    print(NN_layer)
    # path = '/srv/scratch/z5163479/tiny-imagenet-200'
    # dataset = TinyImageNet(path)
    # (train_x, train_y), (test_x, test_y) = dataset.kntl_data_form(k, n, k, n)
    # NN_layer = "Flaten"
    # print("drop out with global pool {}\n".format(k))
    # print("normal flaten {}\n".format(k))

    zero_images = images[labels == 0][:k]
    one_images = images[labels == 1][:k]
    two_images = images[labels == 2][:k]
    three_images = images[labels == 3][:k]
    four_images1 = images[labels == 4][:k]

    # add more data for class 4
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
        four_images2 = add_images[labels == 4][:n]

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
     
    '''
    Split three datasets into X and Y two arrays
        - shuffle each dataset, then split
        - use resnet50 data preprocess method to preprocess X
    '''
    val_x = []
    val_y = []
    random.shuffle(val_test) 
    for features, label in val_test:
        features = preprocess_input(features,  mode='torch', data_format='channels_last')
        val_x.append(features)
        val_y.append(label)
        
    val_x=np.array(val_x).reshape(val_size,587,587,3)
    # val_x = val_x.astype('float32') / 255.0

    train_x = []
    train_y = []
    random.shuffle(img_train)
    for features, label in img_train:
        features = preprocess_input(features,  mode='torch', data_format='channels_last')
        train_x.append(features)
        train_y.append(label)
        
    train_x=np.array(train_x).reshape(train_size,587,587,3)
    # train_x = train_x.astype('float32') / 255.0

    test_x = []
    test_y = []
    random.shuffle(img_test)
    for features, label in img_test:
        features = preprocess_input(features, mode='torch', data_format='channels_last')
        test_x.append(features)
        test_y.append(label)
        
    test_x=np.array(test_x).reshape(test_size,587,587,3)
    # test_x = test_x.astype('float32')/255.0


    # train_y=to_categorical(train_y)
    # test_y=to_categorical(test_y)
    # val_y=to_categorical(val_y)

    '''
    Image augmentation
    '''
    image_gen = ImageDataGenerator(
                                # rotation_range=45,
                                # width_shift_range=0.2,
                                # height_shift_range=0.2,
                                # horizontal_flip=True,
                                # vertical_flip=True,
                                # fill_mode='nearest',
                            #    data_format='channels_last'
                               )
 
    img_gen=image_gen.flow(train_x, train_y, batch_size=BS, shuffle=True)
    # img_gen = create_custom_gen(img_gen)
    # test_datagen = ImageDataGenerator()

    # val_gen=test_datagen.flow(test_x, test_y, batch_size=32, shuffle=False)
    # test_gen=test_datagen.flow(test_x, test_y, batch_size=20, shuffle=False)

    # check_num_each_class(train_y, test_y, val_y)


    model50 = get_model(input_shape=(587,587,3))
    model50.summary()
    adam = optimizers.Adam(lr=0.001)
    model50.compile(optimizer=adam,
                        loss='sparse_categorical_crossentropy',
                        # loss='kullback_leibler_divergence',
                        metrics=['accuracy'])
    # Tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)
    
    '''
    fine-tune some conv layers
    '''
    # # # train_model=model50.fit(train_x, train_y, batch_size=8,epochs=epoch,verbose=1,validation_data=(val_x, val_y), callbacks=[tensorboard])
    # train_model = model50.fit_generator(img_gen, validation_data=(test_x, test_y), epochs=10, steps_per_epoch=len(train_x)//BS, verbose=1)

    # ## start train
    # for layer in model50.layers[:165]:
    #   layer.trainable = False
    # for layer in model50.layers[165:]:
    #   layer.trainable = True

    # model50.compile(optimizer=optimizers.Adam(lr=1e-5)  ,
    #                         # loss='binary_crossentropy',
    #                         loss='sparse_categorical_crossentropy',
    #                     # loss='kullback_leibler_divergence',
    #                     metrics=['accuracy'])
    
    train_model = model50.fit_generator(img_gen, validation_data=(val_x, val_y), epochs=epoch, steps_per_epoch=len(train_x)//BS, verbose=1, callbacks=[tensorboard])

    (loss, accuracy) = model50.evaluate(test_x, test_y, batch_size=64, verbose=1)
    print( 'loss = {:.4f}, accuracy: {:.4f}%'.format(loss,accuracy*100))


    test_pred = model50.predict(test_x, verbose=1, batch_size=64).argmax(axis=1)
    # test_true=test_y.argmax(axis=1) 
    test_true=test_y
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

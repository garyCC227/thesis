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
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU   
from keras import optimizers
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard
import datetime
import imgaug.augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
import sys

sys.stdout = open('./code/adapted_deep_embeddings/Tiny_sigmoid_heavyaug.txt','wt')

# from scipy.ndimage import imread


class TinyImageNet():
    def __init__(self, path):
        self.path = path

        self.images = []
        self.labels = []

    def load_data(self):
        ims, labels = self.load(self.path)

        self.images = self.process_images(ims)
        self.labels = self.process_labels(labels)

        return self.images, self.labels

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        return np.array(labels)

    @classmethod
    def load(cls, path):
        class_id = 0
        id_to_label = {}
        validation_annotations = None
        validation_images = {}
        images = []
        labels = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f == 'val_annotations.txt':
                    validation_annotations = os.path.join(root, f)
                elif f.endswith('.JPEG'):
                    path = os.path.join(root, f)
                    id = f.split('_')[0]
                    if id == 'val':
                        validation_images[f] = path
                    else:
                        if id not in id_to_label:
                            id_to_label[id] = class_id
                            class_id += 1
                        img = image.load_img(path, target_size=(64,64))
                        img = image.img_to_array(img)
                        if len(img.shape) == 2:
                            img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                        images.append(img)
                        labels.append(id_to_label[id])

        with open(validation_annotations) as val_ann:
            for line in val_ann:
                contents = line.split()
                img = image.load_img(validation_images[contents[0]], target_size=(64,64))
                img = image.img_to_array(img)
                if len(img.shape) == 2:
                    img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                images.append(img)
                labels.append(id_to_label[contents[1]])

        return images, labels

    def kntl_data_form(self, k1, n1, k2, n2):
        assert n1 + n2 <= 200
        assert k1 < 550 and k2 < 550
        self.load_data()

        print('Full dataset: {0}'.format(len(self.labels)))

        all_classes = np.unique(self.labels)
        print('Number of classes: {0}'.format(len(all_classes)))

        task2_classes = np.sort(np.random.choice(all_classes, n2, replace=False))
        all_classes = np.delete(all_classes, np.where(np.isin(all_classes, task2_classes)))
        indices = np.isin(self.labels, task2_classes)
        self.x_task2, self.y_task2 = self.images[indices], self.labels[indices]
        shuffle = np.random.permutation(len(self.y_task2))
        self.x_task2, self.y_task2 = self.x_task2[shuffle], self.y_task2[shuffle]

        task1_classes = np.sort(np.random.choice(all_classes, n1, replace=False))
        indices = np.isin(self.labels, task1_classes)
        self.x_task1, self.y_task1 = self.images[indices], self.labels[indices]
        shuffle = np.random.permutation(len(self.y_task1))
        self.x_task1, self.y_task1 = self.x_task1[shuffle], self.y_task1[shuffle]

        # print('Task 1 Full: {0}'.format(len(self.y_task1)))
        # print('Task 2 Full: {0}\n'.format(len(self.y_task2)))

        # Force class labels to start from 0 and increment upwards by 1
        sorted_class_indices = np.sort(np.unique(self.y_task1))
        zero_based_classes = np.arange(0, len(sorted_class_indices))
        for i in range(len(self.y_task1)):
            self.y_task1[i] = zero_based_classes[sorted_class_indices == self.y_task1[i]]

        self.x_train_task1 = []
        self.y_train_task1 = []
        self.x_valid_task1 = []
        self.y_valid_task1 = []

        for i in zero_based_classes:
            all_indices = np.where(self.y_task1 == i)[0]
            idx = np.random.choice(all_indices, k1, replace=False)
            self.x_train_task1.extend(self.x_task1[idx])
            self.y_train_task1.extend(self.y_task1[idx])
            all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
            self.x_valid_task1.extend(self.x_task1[all_indices])
            self.y_valid_task1.extend(self.y_task1[all_indices])

        self.x_train_task1 = np.array(self.x_train_task1)
        self.y_train_task1 = np.array(self.y_train_task1)
        self.x_valid_task1 = np.array(self.x_valid_task1)
        self.y_valid_task1 = np.array(self.y_valid_task1)

        # print('Task 1 training: {0}'.format(len(self.x_train_task1)))
        print('Task 1 validation: {0}'.format(len(self.x_valid_task1)))

        # Force class labels to start from 0 and increment upwards by 1
        sorted_class_indices = np.sort(np.unique(self.y_task2))
        zero_based_classes = np.arange(0, len(sorted_class_indices))
        for i in range(len(self.y_task2)):
            self.y_task2[i] = zero_based_classes[sorted_class_indices == self.y_task2[i]]

        self.x_train_task2 = []
        self.y_train_task2 = []
        for i in zero_based_classes:
            idx = np.random.choice(np.where(self.y_task2 == i)[0], k2, replace=False)
            self.x_train_task2.extend(self.x_task2[idx])
            self.y_train_task2.extend(self.y_task2[idx])
            self.x_task2 = np.delete(self.x_task2, idx, axis=0)
            self.y_task2 = np.delete(self.y_task2, idx, axis=0)

        self.x_train_task2 = np.array(self.x_train_task2)
        self.y_train_task2 = np.array(self.y_train_task2)

        k_test = 550 - k2

        self.x_test_task2 = []
        self.y_test_task2 = []
        for i in zero_based_classes:
            idx = np.random.choice(np.where(self.y_task2 == i)[0], k_test, replace=False)
            self.x_test_task2.extend(self.x_task2[idx])
            self.y_test_task2.extend(self.y_task2[idx])

        self.x_test_task2 = np.array(self.x_test_task2)
        self.y_test_task2 = np.array(self.y_test_task2)

        # print('k = {0}, n = {1}'.format(k2, n2))
        # print('Task 2 training: {0}'.format(len(self.x_train_task2)))
        # print('Task 2 test: {0}\n'.format(len(self.x_test_task2)))

        # return (self.x_train_task1, self.y_train_task1), (self.x_valid_task1, self.y_valid_task1), (self.x_train_task2, self.y_train_task2), (self.x_test_task2, self.y_test_task2)
        return (self.x_train_task2, self.y_train_task2), (self.x_test_task2, self.y_test_task2)


def get_model(input_shape):
  
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
  #for layer in  base_model.layers[:10]:
    #layer.trainable = False
    #layer.padding='same'
 
  #for layer in  base_model.layers[10:]:
    #layer.trainable = True
    #layer.padding='same'
    
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)

  # x = Flatten() (x)
    
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.3)(x)
#   x = Dense(32, activation='relu')(x)
#   x = Dense(128, activation='relu')(x)
  # x = Dropout(0.5)(x)
#   x = Dense(2048, activation='relu')(x)
#   x = Dense(64, activation='relu')(x)
#   x = LeakyReLU(alpha=0.1)(x)
    
#   x = Dropout(0.3)(x)
  #x = Dense(5, activation='softmax')(x)
  #model = Model(base_model.input, x)
  predictions = Dense(5, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
#   for layer in model.layers[:-5]:
#     layer.trainable = False

  return model

def split_data(data_dict):
    trainset = []
    valset = []
    testset=[]
    for label, images in data_dict.items():
        random.shuffle(images)
        img_train, img_test = train_test_split(images, test_size=0.2)
        img_train, img_val = train_test_split(img_train,test_size=0.2)
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
            img = image.load_img(img_path, target_size=(587,587))
            img = image.img_to_array(img)
            img = preprocess_input(img)
            data[label].append([img, label])

    return data

def create_custom_gen(img_gen):
    seq = iaa.Sequential([
        iaa.MultiplyHue((0.5, 1.5))
        # iaa.imgcorruptlike.Contrast(severity=1)
    ])
    for X_batch, y_batch in img_gen:
        hue = seq(images = X_batch.astype(np.uint8))
        yield hue, y_batch

def main():
    classes = n = 5  #TODO:
    
    k= 500 #TODO
    epoch = 70
    NN_layer = "TinyImage_{}classes_2denselayer_normali_{}_epoch{}_imagenet_imgaug".format(classes,k,epoch) #TODO
    BS = 32 #batch size
    print(NN_layer)
    # NN_layer = "Flaten"
    # print("drop out with global pool {}\n".format(k))
    # print("normal flaten {}\n".format(k))

    path = '/srv/scratch/z5163479/tiny-imagenet-200'
    dataset = TinyImageNet(path)
    (train_x, train_y), (test_x, test_y) = dataset.kntl_data_form(k, n, k, n)
    # images = list(zip(X, Y))
    # random.shuffle(images)
    # img_train, img_test = train_test_split(images, test_size=0.2)
    # img_train, val_test = train_test_split(img_train,test_size=0.1)
    
    # print(len(val_test))
    # print(len(img_train))
    # print(len(img_test))
    # val_size = len(val_test)
    # train_size = len(img_train)
    # test_size = len(img_test)
    # # assert val_size + train_size + test_size == k * classes
     
    # val_x = []
    # val_y = []
    # # random.shuffle(val_test)
    # for features, label in val_test:
    #     val_x.append(features)
    #     val_y.append(label)
        
    # val_x=np.array(val_x).reshape(val_size,64,64,3)
    # # val_x = val_x.astype('float32') / 255

    # train_x = []
    # train_y = []
    # # random.shuffle(img_train)
    # for features, label in img_train:
    #     train_x.append(features)
    #     train_y.append(label)
        
    # train_x=np.array(train_x).reshape(train_size,64,64,3)
    # # train_x = train_x.astype('float32') / 255

    # test_x = []
    # test_y = []
    # # random.shuffle(img_test)
    # for features, label in img_test:
    #     test_x.append(features)
    #     test_y.append(label)
        
    # test_x=np.array(test_x).reshape(test_size,64,64,3)
    # # test_x = test_x.astype('float32')/255

    train_y=to_categorical(train_y)
    test_y=to_categorical(test_y)
    # val_y=to_categorical(val_y)


    # new_train_x = []
    # test_x = preprocess_input(test_x)
    # val_x = preprocess_input(val_x)

    image_gen = ImageDataGenerator(
                                # rotation_range=40,
                                # zoom_range=0.2,
                                horizontal_flip= True,
                                vertical_flip=True,
                               data_format='channels_last')
    image_gen.fit(train_x)

    img_gen=image_gen.flow(train_x, train_y, batch_size=BS, shuffle=True)
    # img_gen = create_custom_gen(img_gen)
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

    model50 = get_model(input_shape=(64,64,3))
    model50.summary()
    adam = optimizers.Adam(lr=0.001)
    model50.compile(optimizer=adam,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)

    # train_model=model50.fit(train_x, train_y, batch_size=8,epochs=epoch,verbose=1,validation_data=(val_x, val_y), callbacks=[tensorboard])
    train_model = model50.fit_generator(img_gen, validation_data=(test_x, test_y), epochs=epoch, steps_per_epoch=len(train_x)//BS, verbose=1, callbacks=[tensorboard])

    (loss, accuracy) =  model50.evaluate(test_x, test_y, batch_size=10, verbose=1)
    print( 'loss = {:.4f}, accuracy: {:.4f}%'.format(loss,accuracy*100))


    test_pred = model50.predict(test_x, verbose=1, batch_size=64).argmax(axis=1)
    test_true=test_y.argmax(axis=1) 
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

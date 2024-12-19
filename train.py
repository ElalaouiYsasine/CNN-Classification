import os
import cv2
import numpy as np
from keras.api import layers
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

from fine_tuning import model3

#train
train_path = "../dataset1/train"

x_train=[]

for folder in os.listdir(train_path):
    sub_path = train_path + "/" + folder
    for img in os.listdir(sub_path):
        image_path = os.path.join(sub_path, img)
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_train.append(img_arr)
train_x=np.array(x_train)
train_x=train_x/255.0
train_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(train_path,
                                                    target_size = (180, 180),
                                                    batch_size = 3,class_mode = 'categorical')


train_y=training_set.classes
training_set.class_indices

#validation
val_path = "../dataset1/validation"

x_val=[]

for folder in os.listdir(val_path):
    sub_path = val_path + "/" + folder
    for img in os.listdir(sub_path):
        image_path = os.path.join(sub_path, img)
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_val.append(img_arr)
val_x=np.array(x_val)
val_x=val_x/255.0
train_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(val_path,
                                                 target_size = (180, 180),
                                                 batch_size = 3,
                                                 class_mode = 'categorical')
val_y=training_set.classes
training_set.class_indices
#test

test_path = "../dataset1/test"

x_test=[]

for folder in os.listdir(test_path):
    sub_path = test_path + "/" + folder
    for img in os.listdir(sub_path):
        image_path = os.path.join(sub_path, img)
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_test.append(img_arr)
test_x=np.array(x_test)
test_x=test_x/255.0
train_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(test_path,
                                                 target_size = (180, 180),
                                                 batch_size = 3,
                                                 class_mode = 'categorical')

test_y=training_set.classes
training_set.class_indices
train_y.shape,test_y.shape,val_y.shape

history = model3.fit(
  train_x,
  train_y,
  validation_data=(val_x,val_y), epochs=10,batch_size=3)








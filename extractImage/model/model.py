from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

import cv2

import glob


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def load_data():
    filelist = glob.glob('/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/cropImgDir/*.png')
    _X = []
    _Y = []
    i = 0

    for f in filelist:
        print(i)
        i = i + 1
        file_name = os.path.splitext(os.path.basename(f))[0]
        img_data = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        #print(img_data.shape)
        if img_data is None:
            print("Image {} could not be loaded!!".format(f))
            exit(-1)
        #print(file_name)
        img_resized = cv2.resize(img_data, (72, 72))
        print(img_resized.shape)
        _X.append(img_resized)

        if (file_name.endswith("person")):
            #print("person")
            _Y.append(1)
        if (file_name.endswith("person?")):
            #print("person?")
            _Y.append(2)
        if (file_name.endswith("people")):
            #print("people")
            _Y.append(3)
        if (file_name.endswith("person-fa")):
            #print("people")
            _Y.append(4)
        if(i == 50):
            break

    print("Total in X: {}".format(len(_X)))
    print("Total in Y: {}".format(len(_Y)))

    train_X = np.array(_X)
    train_Y = np.array(_Y)


    print("Shape of train_X: {}".format((train_X.shape)))
    print("Shape of train_Y: {}".format((train_Y.shape)))

#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#print(train_images.shape)

load_data()
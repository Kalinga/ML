from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold

import numpy as np

import os

import cv2

import glob

from timeit import default_timer as timer

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def load_data():
    filelist = glob.glob('/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/resizeImgDir/*.png')
    _X = []
    _Y = []
    i = 0

    for f in filelist:
        #print(i)
        i = i + 1
        file_name = os.path.splitext(os.path.basename(f))[0]
        img_data = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        #print(img_data.shape)
        if img_data is None:
            print("Image {} could not be loaded!!".format(f))
            exit(-1)
        #print(file_name)

        #print(img_data.shape)
        _X.append(img_data)

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
        #if(i == 100000):
        #    break

    print("Total in X: {}".format(len(_X)))
    print("Total in Y: {}".format(len(_Y)))

    train_X = np.array(_X)
    train_Y = np.array(_Y)


    print("Shape of train_X: {}".format((train_X.shape)))
    print("Shape of train_Y: {}".format((train_Y.shape)))
    return (train_X, train_Y)


def kfold_model():
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # load pima indians dataset
    start = timer()
    dataset = load_data()
    end = timer()
    print("load_data() takes {} seconds".format(end - start))

    # split into input (X) and output (Y) variables
    X = dataset[0]
    Y = dataset[1]

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []

    start = timer()
    epochs = [6,7,8]

    for e in epochs:
        print("epoch value {}".format(e))
        for train, test in kfold.split(X, Y):
            # create model
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(72, 72)),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(4, activation=tf.nn.softmax)
            ])

            # Compile model
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            # Fit the model
            #model.fit(train_images, train_labels, epochs=5)
            model.fit(X[train], Y[train], epochs=e, batch_size=3000, verbose=0)
            # evaluate the model
            scores = model.evaluate(X[test], Y[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)
        end = timer()
        print("model fitting takes {} seconds".format(end - start))
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#print(train_images.shape)

#load_data()

kfold_model()
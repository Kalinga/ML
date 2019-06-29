import keras
from keras.models import *


model = load_model("model_entire.h5")
print(type(model))
print(model.summary())
print(model.get_weights())

model.save_weights("model_weights.h5")


model_json = model.to_json()
print(model_json)
from keras.models import model_from_json
model2 = model_from_json(model_json)
model2.load_weights("model_weights.h5")
print(model2.summary())

import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
%matplotlib inline

train_path ="cats-and-dogs/train_path"
valid_path ="cats-and-dogs/valid_path"
test_path  ="cats-and-dogs/test_path"

train_batches = ImageDataGenerator().flow_from_directory(train_path,
    target_size=(224,224), classes=["dog", "cat"], batch_size=10 )
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
    target_size=(224,224), classes=["dog", "cat"], batch_size=4 )
test_batches = ImageDataGenerator().flow_from_directory(test_path,
    target_size=(224,224), classes=["dog", "cat"], batch_size=10 )

def plot(ims, figsize=(12, 6), rows=1, interp = False, titles = None):
    if type(ims[0]) is ndarray

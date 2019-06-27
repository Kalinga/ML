import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)


train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
print(train_samples)
print(train_labels)

scalar = MinMaxScaler(feature_range=(0,1))

#fit_transform does not accept 1D array
scaled_train_sample = scalar.fit_transform((train_samples).reshape(-1, 1))

model = Sequential([
        Dense(16, input_shape=(1,), activation='relu' ), # 16 number of neurons/nodes
        # 32 number of neurons/nodes, default activation function is linear activation function
        Dense(32, activation='relu' ),
        Dense(2,  activation='softmax' ) # last layer, 2 indicates number of class
])

# Adam: Otimizer (SGD, RMSPROP),
model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(scaled_train_sample, train_labels, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=1)

print(model.summary())
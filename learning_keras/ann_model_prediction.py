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

test_labels = []
test_samples = []

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


for i in range(20):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

scalar = MinMaxScaler(feature_range=(0,1))

#fit_transform does not accept 1D array
scaled_train_sample = scalar.fit_transform((train_samples).reshape(-1, 1))
scaled_test_samples = scalar.fit_transform((test_samples).reshape(-1, 1))

model = Sequential([
        Dense(16, input_shape=(1,), activation='relu' ), # 16 number of neurons/nodes
        # 32 number of neurons/nodes, default activation function is linear activation function
        Dense(32, activation='relu' ),
        Dense(2,  activation='softmax' ) # last layer, 2 indicates number of class
])

# Adam: Otimizer (SGD, RMSPROP),
model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(scaled_train_sample, train_labels, validation_split=0.1,
          batch_size=10, epochs=20, shuffle=True, verbose=1)

prediction = model.predict(scaled_test_samples, batch_size=10)
for i in prediction:
    print(i)

prediction_rounded = model.predict_classes(scaled_test_samples, batch_size=10)
for i in prediction_rounded:
    print(i)

print(model.summary())

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, prediction_rounded)
cm_plot_labels = ["no side effects", "side effects"]

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #return ax

plot_confusion_matrix(cm, cm_plot_labels )

model.save("model_entire.h5")
model.save_weights("model_weight.h5")

np.set_printoptions(precision=2)

plt.show()

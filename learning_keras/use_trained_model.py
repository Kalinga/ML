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

# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
plt.imshow(ims[i], interpolation=None if interp else 'none')

imgs, labels = next(train_batches)
plots(imgs, titles=labels)

model = Sequential([
    # 32 is the number of output filters in the convulation
    # (3,3) kernel size, convulational window,
    # input must be specified for the first layer of the any sequential layer
    # Flatten() transforming the output of the previous layer to a 1D array
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)), # convulational layer
    Flatten(),
    Dense(2, activation='softmax')
    ])

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, # Image Data generator generates batch by batch
                    steps_per_epoch=4, # total number of data / batch size
                    validation_data=valid_batches, # Image Data generator generates validation batch by batch
                    validation_steps=4,
                    epochs=5, # Total number of epocs for taraining
                    verbose=2)

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

#Predict
test_imgs,test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)

print(test_labels.shape())
test_labels = test_labels[,:0]
print(test_labels)

prediction = model.predict_generator(test_batches, steps=1, verbose=0)

cm_plot_labels=['cat', 'dog']
cm = confusion_matrix(test_labels, prediction[:,0])
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
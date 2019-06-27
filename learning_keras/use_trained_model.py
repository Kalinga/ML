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


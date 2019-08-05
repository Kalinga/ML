from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
#from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

from models.keras_ssd7 import build_model

# Set a few configuration parameters.
img_height = 480 # Height of the input images
img_width = 640 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 1 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.1, 0.2, 0.33, 0.413, 0.418, 0.8] #width/height
#aspect_ratios = [0.5, 1.0, 2.0] # The ldist of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
model_mode = 'inference'


K.clear_session() # Clear previous models from memory.

#Loss function
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# model = build_model(image_size=(img_height, img_width, img_channels),
#                     n_classes=n_classes,
#                     mode=model_mode,
#                     l2_regularization=0.0005,
#                     scales=scales,
#                     aspect_ratios_global=aspect_ratios,
#                     aspect_ratios_per_layer=None,
#                     two_boxes_for_ar1=two_boxes_for_ar1,
#                     steps=steps,
#                     offsets=offsets,
#                     clip_boxes=clip_boxes,
#                     variances=variances,
#                     normalize_coords=normalize_coords,
#                     subtract_mean=intensity_mean,
#                     divide_by_stddev=intensity_range)

model_path = '/home/kara9147/ML/ssd_keras_caltech/ssd7_epoch-03_loss-2.4693_val_loss-2.4097.h5'

print("Using saved model: {}".format(model_path))
model = load_model(model_path,
                   custom_objects={'AnchorBoxes': AnchorBoxes, 'compute_loss': ssd_loss.compute_loss})

#model.load_weights(model_path)

# Images
images_dir = '/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/images/'

# Ground truth
val_labels_filename   = '/home/kara9147/ML/ssd_keras_caltech/labels_val.csv'

val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path="./dataset_caltech_val.h5")
val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all'
                      #include_classes=[1, 2]
                      )

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=val_dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=1,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results
print (mean_average_precision)
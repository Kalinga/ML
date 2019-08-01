
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation


import numpy as np
import cv2
import time

# https://github.com/farrajota/caltech-extract-data-toolbox/blob/master/vbb.m

model = None

# For Caltech Data
img_height = 480  # Height of the input images
img_width = 640  # Width of the input images
img_channels = 3  # Number of color channels of the input images
intensity_mean = 127.5  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 1  # Number of positive classes pepole, person, person-fa, person?
scales = [0.08, 0.16, 0.32, 0.64,
          0.96]  # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.1, 0.2, 0.33, 0.413, 0.418, 0.8]#width/height
#aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True  # Whether or not the model is supposed to use coordinates relative to the image size

def build_prediction_model():
    global model

    # 1: Build the Keras model

    K.clear_session()  # Clear previous models from memory.

    # Loss function
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model_path = '/home/kara9147/ML/ssd_keras_caltech/ssd7_epoch-01_loss-3.0443_val_loss-2.9963.h5'


    # 2: Load the saved model

    model = load_model(model_path,
                      custom_objects={'AnchorBoxes': AnchorBoxes, 'compute_loss': ssd_loss.compute_loss})

def play():
    start_time_video = time.time()
    #cap = cv2.VideoCapture("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/set00_V000.avi")
    cap = cv2.VideoCapture("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/set03_V008.avi")
    #cap = cv2.VideoCapture("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/set08_V004.avi")

    # Time to read all frames, predict and put bounding boxes around them, and show them.
    i = 0
    total_time = 0.0

    # Capture frame-by-frame
    ret = True
    while(ret):
        ret, img = cap.read()
        i = i + 1
        #print("Processing {} th frame".format(i))
        if (ret != False):
            # Our operations on the frame come here

            # Open CV uses BGR color format
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #print(type(img))
            #print(img.shape)

            batch_img = np.expand_dims(frame , axis=0)
            #print(batch_img.shape )

            current = time.time()
            ##################################PREDICTION######################
            y_pred = model.predict(batch_img)
            end = time.time()
            diff = end - current
            total_time  = total_time  + diff
            print(end - current)
            print("Time spent for predicting: {0}".format(diff))

            # 4: Decode the raw prediction `y_pred`

            y_pred_decoded = decode_detections(y_pred,
                                               confidence_thresh=0.5,
                                               iou_threshold=0.4,
                                               top_k=200,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width)

            np.set_printoptions(precision=2, suppress=True, linewidth=90)
            # print("Predicted boxes:\n")
            # print('   class   conf xmin   ymin   xmax   ymax')
            print(y_pred_decoded)

            #print(time.time() - start_time)

            if (y_pred_decoded and len(y_pred_decoded[0])):
                colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()  # Set the colors for the bounding boxes
                classes = ['background', 'person', 'people']  # Just so we can print class names onto the image instead of IDs


                # Draw the predicted boxes in blue
                for box in y_pred_decoded[0]:
                    xmin = int(box[-4])
                    ymin = int(box[-3])
                    xmax = int(box[-2])
                    ymax = int(box[-1])
                    color = colors[int(box[0])]
                    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])

                    #print((xmin, ymin))
                    #print((xmax, ymax))

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax ), (255, 0, 0), 1)

            # Display the resulting frame
            cv2.imshow('frame',img)

        # waitKey: 0, wait indefinitely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time_video = time.time()
    print("No of frames: {}".format(i))
    print("Total Time: {}".format(total_time))
    print("fps: {}".format(i / (total_time)))



    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

build_prediction_model()
play()
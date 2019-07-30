
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

    fresh_model = False

    # 1: Build the Keras model

    K.clear_session()  # Clear previous models from memory.

    # Loss function
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model = build_model(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=scales,
                        aspect_ratios_global=aspect_ratios,
                        aspect_ratios_per_layer=None,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=intensity_mean,
                        divide_by_stddev=intensity_range)

    # 2: Optional: Load some weights

    # model.load_weights('./ssd7_weights.h5', by_name=True)
    # model.load_weights('./ssd7_epoch-15_loss-2.1809_val_loss-2.3229.h5', by_name=True)
    if fresh_model:
        # 3: Instantiate an Adam optimizer and the SSD loss function (moved out!) and compile the model

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    else:
        model_path = '/home/kara9147/ML/ssd_keras_caltech/1class_epoch-03_loss-2.4099_val_loss-2.4834.h5'

        print("Using saved model: {}".format(model_path))
        model = load_model(model_path,
                           custom_objects={'AnchorBoxes': AnchorBoxes, 'compute_loss': ssd_loss.compute_loss})

def useless_code():
        # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

        # Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

        # train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path="dataset_caltech_train.h5")
        train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path="dataset_caltech_train.h5")
        val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path="dataset_caltech_val.h5")

        # 2: Parse the image and label lists for the training and validation datasets.

        # TODO: Set the paths to your dataset here.

        # Images
        images_dir = '/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/images/'

        # Ground truth
        train_labels_filename = '/home/kara9147/ML/ssd_keras_caltech/labels_train.csv'
        val_labels_filename = '/home/kara9147/ML/ssd_keras_caltech/labels_val.csv'

        train_dataset.parse_csv(images_dir=images_dir,
                                labels_filename=train_labels_filename,
                                input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                                # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                                include_classes='all')

        val_dataset.parse_csv(images_dir=images_dir,
                              labels_filename=val_labels_filename,
                              input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                              include_classes='all')

        # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
        # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
        # option in the constructor, because in that cas the images are in memory already anyway. If you don't
        # want to create HDF5 datasets, comment out the subsequent two function calls.

        createhdf5 = False
        if createhdf5:
            # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
            # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
            # option in the constructor, because in that case the images are in memory already anyway. If you don't
            # want to create HDF5 datasets, comment out the subsequent two function calls.

            train_dataset.create_hdf5_dataset(file_path='dataset_caltech_train.h5',
                                              resize=False,
                                              variable_image_size=True,
                                              verbose=True)

            val_dataset.create_hdf5_dataset(file_path='dataset_caltech_val.h5',
                                            resize=False,
                                            variable_image_size=True,
                                            verbose=True)

        # Get the number of samples in the training and validations datasets.
        train_dataset_size = train_dataset.get_dataset_size()
        val_dataset_size = val_dataset.get_dataset_size()

        print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
        print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

        # 1: Set the generator for the predictions.

        predict_generator = val_dataset.generate(batch_size=1,
                                                 shuffle=False,
                                                 transformations=[],
                                                 label_encoder=None, #ssd_input_encoder,
                                                 returns={'processed_images',
                                                          'processed_labels',
                                                          'filenames'},
                                                 keep_images_without_gt=False)

def predict(img):
    # 2: Generate samples

    # batch_images, batch_labels, batch_filenames = next(predict_generator)
    # print(len(batch_images))
    #
    # #i Which batch item to look at
    # for  i in range(len(batch_filenames)):
    #     print("Image:", batch_filenames[i])
    #     print("Ground truth boxes:\n")
    #     for  c in range(len(batch_labels[i])):
    #         print(batch_labels[i][c])

    # 3: Make a prediction
    start = time.time()
    print(start)
    y_pred = model.predict(img)
    end = time.time()
    print(end)
    print("Predict time: ".format(end - start))

    # 4: Decode the raw prediction `y_pred`

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.25,
                                       iou_threshold=0.1,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    #print("Predicted boxes:\n")
    #print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded)
    return y_pred_decoded

def draw():
    # 5: Draw the predicted boxes onto the image

    #plt.figure(figsize=(20,12))
    #plt.imshow(batch_images[i])
    plt.figure(figsize=(20,12))
    plt.imshow(batch_images[i])
    current_axis = plt.gca()

    colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
    classes = ['background', 'person', 'people', 'person-fa', 'person?'] # Just so we can print class names onto the image instead of IDs

    # Draw the ground truth boxes in green (omit the label for more clarity)
    for box in batch_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
        #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

    # Draw the predicted boxes in blue
    for box in y_pred_decoded[i]:
        xmin = box[-4]
        ymin = box[-3]
        xmax = box[-2]
        ymax = box[-1]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

def play():
    start_time_video = time.time()
    cap = cv2.VideoCapture("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/set00_V000.avi")
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
            y_pred = model.predict(batch_img)
            end = time.time()
            diff = end - current
            total_time  = total_time  + diff
            print(end - current)
            print("Time spent for predicting: {0}".format(diff))

            # 4: Decode the raw prediction `y_pred`

            y_pred_decoded = decode_detections(y_pred,
                                               confidence_thresh=0.5,
                                               iou_threshold=0.1,
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
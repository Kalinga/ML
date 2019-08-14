from keras import backend as K
from keras.models import load_model

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes

from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

# Set a few configuration parameters.
img_height = 480 # Height of the input images
img_width = 640 # Width of the input images

n_classes = 1 # Number of positive classes

model_mode = 'training'


K.clear_session() # Clear previous models from memory.

#Loss function
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)


model_path = '/home/kara9147/ML/ssd_keras_caltech/ssd7_epoch-03_loss-2.4693_val_loss-2.4097.h5'

print("Using saved model: {}".format(model_path))
model = load_model(model_path,
                   custom_objects={'AnchorBoxes': AnchorBoxes, 'compute_loss': ssd_loss.compute_loss})

# Images
images_dir = '/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/images/'

# Ground truth
val_labels_filename  = '/home/kara9147/ML/ssd_keras_caltech/labels_test.csv'

h5 = False

if h5:
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path="./dataset_caltech_test.h5")
else:
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    val_dataset.parse_csv(images_dir=images_dir,
                          labels_filename=val_labels_filename,
                          input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                          include_classes='all'
                          #include_classes=[1, 2]
                          )

    val_dataset.create_hdf5_dataset(file_path='./dataset_caltech_test.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)

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
                    #decoding_iou_threshold = 0.1,
                    #decoding_top_k = 200,
                    #decoding_normalize_coords = True,
                    #decoding_confidence_thresh = 0.1,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results
print (mean_average_precision)
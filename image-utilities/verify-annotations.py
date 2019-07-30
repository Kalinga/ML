import cv2
import csv

def bb():
    img_path = "/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/images/"
    train_data_file = "/home/kara9147/ML/ssd_keras_caltech/labels_val.csv"

    reader = csv.DictReader(open( train_data_file))
    i = 0
    for raw in reader:
        print( img_path + raw["frame"], raw["xmin"], raw["xmax"], raw["ymin"], raw["ymax"], raw["class_id"] )

        boudingBoxing(img_path + raw["frame"],
                      int(raw["xmin"]),
                      int(raw["ymin"]),
                      int(raw["xmax"]),
                      int(raw["ymax"]))
        if (i == 1000):
            exit(0)
        i = i +1

def boudingBoxing(path, x_min, y_min, x_max, y_max):
    print(path,  x_min, y_min, x_max, y_max)

    img = cv2.imread(path, 0)
    if img is None:
        print("Image could not be loaded!!")
        exit(-1)
    print(img.shape)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0, 1))

    cv2.imshow('image', img)
    cv2.waitKey(5000)
    #cv2.destroyAllWindows()

#print (cv2.getBuildInformation())

def confirm_csv_format():
    # First, just read in the CSV file lines and sort them.

    data = []
    labels_filename = "/home/kara9147/ML/ssd_keras_caltech/labels_val.csv"
    input_format = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']
    labels_output_format = ['xmin', 'ymin', 'xmax', 'ymax', 'class_id']

    include_classes = 'all'
    with open(labels_filename, newline='') as csvfile:
        csvread = csv.reader(csvfile, delimiter=',')
        next(csvread)  # Skip the header row.
        for row in csvread:  # For every line (i.e for every bounding box) in the CSV file...
            if include_classes == 'all' or int(row[input_format.index(
                    'class_id')].strip()) in include_classes:  # If the class_id is among the classes that are to be included in the dataset...
                box = []  # Store the box class and coordinates here
                box.append(row[input_format.index(
                    'image_name')].strip())  # Select the image name column in the input format and append its content to `box`
                for element in labels_output_format:  # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                    box.append(int(row[input_format.index(
                        element)].strip()))  # ...select the respective column in the input format and append it to `box`.
                data.append(box)

    data = sorted(data)  # The data needs to be sorted, otherwise the next step won't give the correct result
    print(data)

boudingBoxing("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/images/set00_V001_974.png", 528, 103, 548, 376)
#boudingBoxing("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/images/set00_V001_974.png", 450,111,584,426)

#bb()
#confirm_csv_format()
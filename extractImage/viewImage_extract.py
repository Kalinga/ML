import json
import os
import cv2
import glob
from matplotlib import pyplot as plt

imgDir = "/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/images/"
cropImgDir = "/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/cropImgDir/"
annotations = json.load(open('/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/annotations.json'))
set_name = "set05"
#video_name = ["V001", "V002", "V003", "V004", "V005", "V006", "V007", "V008", "V009", "V010", "V011", "V012", "V013", "V014"]
video_name = ["V001", "V002", "V003", "V004", "V005", "V006", "V007", "V008", "V009", "V010", "V011", "V012"]

def allVideos():
    for v in video_name:
        print(v)

def estimateMaxWidthAndHeight():
    w_arr = []
    h_arr = []

    labeledFrames = annotations[set_name][video_name]['frames']

    print(type(labeledFrames))
    for key, value in labeledFrames.items():
        print(key)

        frameDetails = annotations[set_name][video_name]['frames'][key]
        for idx, items in enumerate(frameDetails):
            print(key, ": ", idx, ": ", items, "\n")
            img_file_name = "{}_{}_{}_{}_{}.png".format(set_name, video_name, key, idx + 1, items['lbl'])
            print(img_file_name)
            pos = items['pos']
            print(pos)

            # [l t w h]

            l = int(pos[0])
            t = int(pos[1])
            w = int(pos[2])
            h = int(pos[3])

            w_arr.append(w)
            h_arr.append(h)

            print(l, t, w, h)
    print("max width.", max(w_arr))
    print("max height.", max(h_arr))

def cropAndSave():
    for v in video_name:
        labeledFrames = annotations[set_name][v]['frames']

        print(type(labeledFrames))
        for key, value in labeledFrames.items():
            print(key)
            fileName = "{}_{}_{}.png".format(set_name, v, key)
            print(fileName)
            img = cv2.imread(imgDir + fileName, 0)
            if img is None:
                print("Image could not be loaded!!")
                exit(-1)

            frameDetails = annotations[set_name][v]['frames'][key]
            for idx, items in enumerate(frameDetails):
                print(key, ": ", idx, ": ", items, "\n")
                img_file_name = "{}_{}_{}_{}_{}.png".format(set_name, v, key, idx + 1, items['lbl'])
                print(img_file_name)
                pos = items['pos']
                print(pos)

                # [l t w h]

                l = int(pos[0])
                t = int(pos[1])
                w = int(pos[2])
                h = int(pos[3])

                print(l, t, w, h)

                crop_img = img[t:t + h, l:l + w].copy()
                cv2.imwrite(cropImgDir + img_file_name, crop_img)



def boudingBoxing():
    fileName = "set00_V000_390.png"

    labeledFrames = annotations[set_name][video_name[0]]['frames']

    img = cv2.imread(imgDir + fileName, 0)
    if img is None:
        print("Image could not be loaded!!")
        exit(-1)
    print(img.shape)

    frameDetails = annotations[set_name][video_name[0]]['frames']['390']

    w_arr = []
    h_arr = []
    for idx, items in enumerate(frameDetails):
        img_file_name = "{}_{}_{}_{}_{}.bmp".format(set_name, video_name[0], '390', idx + 1, items['lbl'])
        print(img_file_name)
        pos = items['pos']
        print(pos)
        # [l t w h]

        l = int(pos[0])
        t = int(pos[1])
        w = int(pos[2])
        h = int(pos[3])

        w_arr.append(w)
        h_arr.append(h)

        print(l, t, w, h)

        # crop_img = img[int(pos[1]):int(pos[1]) + int(pos[2]), int(pos[0]):int(pos[0]) + int(pos[3])]
        #crop_img = img[t:t+h, l:l+w]
        cv2.rectangle(img, (l, t), (l + w, t + h), (0, 0, 255), 1)
        # cv2.imshow("cropped", img)
        # cv2.waitKey(0)

    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def resize_store_gray_images():
    filelist = glob.glob('/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/cropImgDir/*.png')

    for f in filelist:
        file_name = os.path.basename(f)
        img_data = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        #print(img_data.shape)
        if img_data is None:
            print("Image {} could not be loaded!!".format(f))
            exit(-1)
        print(file_name)
        img_resized = cv2.resize(img_data, (72, 72))
        cv2.imwrite(resizeImgDir + file_name, img_resized)

#estimateMaxWidthAndHeight()
#cropAndSave()
#allVideos()
#boudingBoxing()
resize_store_gray_images()



#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
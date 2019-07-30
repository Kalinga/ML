import json
import os
import cv2
import glob
from matplotlib import pyplot as plt

imgDir = "/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/images/"
bbImgDir = "/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/bbImgDir/"
annotations = json.load(open('/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/annotations.json'))
set_name = "set05"
#video_name = ["V001", "V002", "V003", "V004", "V005", "V006", "V007", "V008", "V009", "V010", "V011", "V012", "V013", "V014"]
video_name = ["V001", "V002", "V003", "V004", "V005", "V006", "V007", "V008", "V009", "V010", "V011", "V012"]

def bb_save():
    i = 0
    v = 0
    while(i < 5):
        print(video_name[v])
        print("----------------------------------")
        labeledFrames = annotations[set_name][video_name[v]]['frames']
        for key, value in labeledFrames.items():
            #print(key)
            frameDetails = annotations[set_name][video_name[v]]['frames'][key]
            for idx, items in enumerate(frameDetails):
                #print(key, ": ", idx, ": ", items, "\n")
                if (items['lbl'] == "person-fa"):
                    print(v, i, video_name[v])
                    print("person-fa")
                    img_file_name = "{}_{}_{}_{}_{}.png".format(set_name, video_name[v], key, idx + 1, items['lbl'])

                    fileName = "{}_{}_{}.png".format(set_name, video_name[v], key)
                    print(fileName)
                    img = cv2.imread(imgDir + fileName, 0)

                    pos = items['pos']
                    print(pos)

                    # [l t w h]

                    l = int(pos[0])
                    t = int(pos[1])
                    w = int(pos[2])
                    h = int(pos[3])
                    cv2.rectangle(img, (l, t), (l + w, t + h), (0, 0, 255), 1)
                    cv2.imwrite(bbImgDir + img_file_name, img)
                    i = i + 1
        v = v + 1


bb_save()
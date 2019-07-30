import json
import numpy as np
import cv2

def annotationSchema():
    annotations = json.load(open('/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/annotations.json'))
    set_name = "set00"
    video_name = "V000"

    data = annotations[set_name][video_name]
    for key, value in data.items():
        print(key)


    print("1.altered {}:".format(annotations[set_name][video_name]['altered']))
    print("2.logLen {}:".format(annotations[set_name][video_name]['logLen']))
    print("3.log {}:".format(annotations[set_name][video_name]['log']))
    print("4.maxObj {}:".format(annotations[set_name][video_name]['maxObj']))
    print("5. nFrame {}:".format(annotations[set_name][video_name]['nFrame']))
    print("6. frames : List of frames.. ")

    frames = annotations[set_name][video_name]['frames']
    #if frames is None:
    #    print("No frames")
    #else:
       # print("No of frames in the Vid {}".format(len(data)))
    #print(frames)

    frameNo =[]
    for key, value in frames.items():
        frameNo.append(key)
    print(sorted(frameNo))
    print(len(frameNo))

    print("individual frame details : annotations[set_name][video_name]['frames']['xxx']")

    # {'person', 'person-fa', 'people'}
    #classes = set()
    #for key, value in frames.items():
    #    frameDetails = annotations[set_name][video_name]['frames'][key]
    #    for idx, items in enumerate(frameDetails):
    #        #print(idx, ": \n", items['lbl'])
    #        classes.add(items['lbl'])
    #print(classes)

    for key, value in frames.items():
        frameDetails = annotations[set_name][video_name]['frames'][key]
        for idx, items in enumerate(frameDetails):
            print(key, ": ", idx, ": ", items, "\n")
            img_file_name = "{}_{}_{}_{}_{}.bmp".format(set_name, video_name, key, idx, items['lbl'])
            print(img_file_name)




#for key, value in data.items():
#    print(key)

def play():
    path = "/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/"
    set_name = "set00"
    video_name = "V006"

    annotations = json.load(open('/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/annotations.json'))
    frames = annotations[set_name][video_name]['frames']
    print(frames)


    cap = cv2.VideoCapture(path + set_name + "_" + video_name + ".avi")
    imageFileName = ""
    frameNo = 1
    while(cap.isOpened()):
        ret, frame = cap.read()

        try :
            frame_details = annotations[set_name][video_name]['frames'][str(frameNo)]
            if frame_details is not None:
                print("Details for {}".format(frameNo))
                print(frame_details)
                #for key, value in frame_details.items():
                #    print(key)
        except KeyError:
            print("No details for {}". format(frameNo))

        frameNo = frameNo + 1

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',frame)
        if cv2.waitKey(250) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#annotationSchema()
play()




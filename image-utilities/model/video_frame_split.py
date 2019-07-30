import numpy as np
import cv2

cap = cv2.VideoCapture("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/set07_V011.avi")

import timeit

start = timeit.timeit()

# Time to read all frames, predict and put bounding boxes around them, and show them.
i = 0

# Capture frame-by-frame
ret = True
while(ret):
    ret, frame = cap.read()
    i = i + 1
    if (ret != False):
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)

    # waitKey: 0, wait indefinitely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("No of frames: {}".format(i))

end = timeit.timeit()

print(start)
print(end)
print (end - start)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def allComb():
    path = "/home/kara9147/CODE/Images/"
    imgPath = path + "kashvi.jpg"
    img = cv2.imread(imgPath, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Matrix to be used as kernel in the Convolution process
    # Identity
    k = np.array(([0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]),
                 np.float32)
    print(k)
    output_unchanged  = cv2.filter2D(img, -1, k)

    # using np
    k_allOnes = np.array(np.ones((3, 3), np.float32))
    output_with_All_ones  = cv2.filter2D(img, -1, k_allOnes)

    k_allZeros = np.array(np.zeros((3, 3), np.float32))
    output_with_All_zeros = cv2.filter2D(img, -1, k_allZeros)

    k_edge_detect_1 =  np.array(
                ([ 1,  0, -1],
                 [ 0,  0,  0],
                 [-1,  0,  1]),
                 np.float32)

    output_k_edge_detect_1 = cv2.filter2D(img, -1, k_edge_detect_1)

    k_edge_detect_2 = np.array(
        ([0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]),
        np.float32)
    output_k_edge_detect_2 = cv2.filter2D(img, -1, k_edge_detect_2)

    k_edge_detect_3 = np.array(
        ([-1, -1, -1],
         [-1,  8, -1],
         [-1, -1, -1]),
        np.float32)
    output_k_edge_detect_3 = cv2.filter2D(img, -1, k_edge_detect_3)

    k_sharpen = np.array(
        ([0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]),
        np.float32)
    output_k_sharpen  = cv2.filter2D(img, -1, k_sharpen )

    plt.figure(figsize=(100,300))
    plt.tight_layout()
    plt.subplots_adjust(wspace=4)
    plt.subplot(4, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")

    plt.subplot(4, 2, 2)
    plt.imshow(output_unchanged)
    plt.title("Filtered Unchanged Image")

    plt.subplot(4, 2, 3)
    plt.imshow(output_with_All_ones)
    plt.title("Filtered All Ones Image")

    plt.subplot(4, 2, 4)
    plt.imshow(output_with_All_zeros)
    plt.title("Filtered All Zeros Image")

    plt.subplot(4, 2, 5)
    plt.imshow(output_k_edge_detect_1)
    plt.title("Filtered k_edge_detect_1")

    plt.subplot(4, 2, 6)
    plt.imshow(output_k_edge_detect_2)
    plt.title("Filtered k_edge_detect_2")

    plt.subplot(4, 2, 7)
    plt.imshow(output_k_edge_detect_3)
    plt.title("Filtered k_edge_detect_3")

    plt.subplot(4, 2, 8)
    plt.imshow(output_k_sharpen)
    plt.title("Filtered k_sharpen")

    plt.show()

def edge():
    path = "/home/kara9147/CODE/Images/"
    imgPath = path + "kashvi.jpg"
    img = cv2.imread(imgPath, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    k_edge_detect_2 = np.array(
        ([0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]),
        np.float32)
    output_k_edge_detect_2 = cv2.filter2D(img, -1, k_edge_detect_2)

    k_edge_detect_3 = np.array(
        ([-1, -1, -1],
         [-1,  8, -1],
         [-1, -1, -1]),
        np.float32)
    output_k_edge_detect_3 = cv2.filter2D(img, -1, k_edge_detect_3)

    plt.subplot(2, 1, 1)
    plt.imshow(output_k_edge_detect_2)
    plt.title("output_k_edge_detect_2")

    plt.subplot(2, 1, 2)
    plt.imshow(output_k_edge_detect_3)
    plt.title("output_k_edge_detect_3")

    plt.show()

if __name__ == '__main__':
    allComb()
    #edge()
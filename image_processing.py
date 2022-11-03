import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage.measure._find_contours


image_dir='Datasets/Data/test/adenocarcinoma/000109 (2).png'
image=cv2.imread(image_dir)
image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


processed_image=image

def display(image,processed):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(processed,cmap='gray')
    plt.title('Processed Image')

    plt.show()

def show_slice_window(slice, level, window):
    """
    Function to display an image slice
    Input is a numpy 2D array
    """
    og=slice
    max = level + window/2
    min = level - window/2
    slice = slice.clip(min,max)
    print(np.max(slice))
    display(og,slice)
    # plt.savefig('L'+str(level)+'W'+str(window))


def canny_edge_detection():
    # Setting parameter values
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold

    # Applying the Canny Edge filter
    edge = cv2.Canny(image, t_lower, t_upper)
    pro=cv2.addWeighted(image,0.7, edge, 0.3, 0)
    display(image, pro)

# show_slice_window(image,100,160)
canny_edge_detection()



import matplotlib.pyplot as plt
import cv2
from skimage.io import imread_collection
import numpy as np


def display(image,processed):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(processed,cmap='gray')
    plt.title('Processed Image')

    plt.show()


image='Datasets/Data/test/adenocarcinoma/000108 (3).png'

img=cv2.imread(image)

blurred_image=cv2.GaussianBlur(img,(5,5),0)
processed=img-blurred_image
sharpened=img+processed
display(img,sharpened)




import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image(filepath):
    imageBGR = cv2.imread(filepath, 1)
    imageRGB = BGR2RGB(imageBGR)
    return imageRGB


def show_image(image,gray=False, name=None, save=False):
    if gray is True:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        plt.imshow(image)
    if save is True:
        plt.savefig(f'{name}.png')
    plt.show()


def BGR2RGB(image):
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    new_image = np.dstack((red, green, blue))
    return new_image

def convert2gray(image):
    res = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return res

def gaussianblur(image, std):
    size = (std * 4 + 1, std * 4 + 1)
    res = cv2.GaussianBlur(image, size, std)
    return res
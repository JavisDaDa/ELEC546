import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.ndimage
import scipy.misc
from PIL import Image

def load_image(filepath):
    imageBGR = cv2.imread(filepath, 1)
    imageRGB = BGR2RGB(imageBGR)
    return imageRGB


def crop_head(image):
    res = image[200:450, 260:450, :]
    return res

def RGBcomponent(image, RGB='G'):
    RGB = RGB.upper()
    blue = image[:, :, 2]
    green = image[:, :, 1]
    red = image[:, :, 0]
    image_b = np.dstack((np.zeros(blue.shape, np.uint8), np.zeros(blue.shape, np.uint8), blue))
    image_g = np.dstack((np.zeros(green.shape, np.uint8), green, np.zeros(green.shape, np.uint8)))
    image_r = np.dstack((red, np.zeros(red.shape, np.uint8), np.zeros(red.shape, np.uint8)))
    if RGB == 'G':
        return image_g
    elif RGB == 'B':
        return image_b
    elif RGB == 'R':
        return image_r
    elif RGB == 'RGB':
        return (image_r, image_g, image_b)
    else:
        raise ValueError('You should enter R, G, B or RGB')


def show_image(image,gray=False, name=None, save=False):
    if gray is True:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        plt.imshow(image)
    if save is True:
        imageio.imsave(f'{name}.png', image)
        #plt.savefig(f'{name}.png')
    plt.show()


def RGB2GRB(image):
    blue = image[:, :, 2]
    green = image[:, :, 1]
    red = image[:, :, 0]
    new_image = np.dstack((green, red, blue))
    return new_image


def BGR2RGB(image):
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    new_image = np.dstack((red, green, blue))
    return new_image


def convert2gray(image):
    res = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return res


def plot_hist(image, name=None, save=False):
    plt.hist(image.ravel(), 50, [0, 256])
    if save is True:
        plt.savefig(f'{name}.png')
    plt.show()

def gaussianblur1(image, std):
    res = cv2.GaussianBlur(image, (15, 15), std)
    return res
def gaussianblur(image, std):
    size = (std*4+1, std*4+1)
    res = cv2.GaussianBlur(image, size, std)
    return res


def matrix_filter(matrix, fil, flag=0):
    m, n = fil.shape[0], fil.shape[1]
    mm, nn = matrix.shape[0], matrix.shape[1]
    res = np.zeros((mm, nn))
    for i in range(m//2, mm-m//2):
        for j in range(n//2, nn-n//2):
            if flag == 0:
                array = np.array([matrix[i, j - 1], matrix[i, j], matrix[i, j+1]]).reshape(m, n)
                num = array*fil
                num = np.sum(num, axis=1)
                res[i, j] = np.round(num)
            elif flag == 1:
                array = np.array([matrix[i - 1, j], matrix[i, j], matrix[i + 1, j]]).reshape(m, n)
                num = array * fil
                num = np.sum(num, axis=0)
                res[i, j] = np.round(num)
            elif flag == 2:
                array = np.array([matrix[i - 1, j - 1], matrix[i - 1, j], matrix[i - 1, j + 1],
                                  matrix[i, j - 1], matrix[i, j], matrix[i, j + 1],
                                  matrix[i + 1, j - 1], matrix[i + 1, j], matrix[i + 1, j + 1]]).reshape(m, n)
                num = array*fil
                num = np.sum(np.sum(num, axis=0))
                res[i, j] = np.round(num)
            else:
                raise ValueError('Please enter 0, 1 or 2')
    return res


def average_filter(image, size):
    res = cv2.blur(image, size)
    return res

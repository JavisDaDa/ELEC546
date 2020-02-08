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


def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling

    Args:
        img_in       (ndarray) monochrome input image
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
        reg_constant (float)   optional regularization constant for pathalogical cases

    Returns:
        result       (ndarray) output bilateral-filtered image

    Raises:
        ValueError whenever img_in is not a 2D float32 valued numpy.ndarray
    """

    # check the input
    if not isinstance( img_in, np.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
        raise ValueError('Expected a 2D numpy.ndarray with float32 elements')

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and
    # the unnormalized result image
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum
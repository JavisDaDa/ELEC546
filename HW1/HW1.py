import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2
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
        plt.savefig(f'{name}.png')
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


def gaussianblur(image, std):
    res = cv2.GaussianBlur(image, (15, 15), std)
    return res


def matrix_filter(matrix, fil):
    m, n = fil.shape[0], fil.shape[1]
    print(m)
    print(n)
if __name__ == '__main__':
    # path = 'Iron_man.jpg'
    # image = load_image(path)
    # show_image(image)
    # #1.1
    # image_head = crop_head(image)
    # #1.2
    # show_image(image_head, name='iron_man_head', save=True)
    # #1.3
    # image_green = RGBcomponent(image_head, RGB='G')
    # show_image(image_green, name='iron_man_head_green', save=True)
    # #1.4
    # GRB_image = RGB2GRB(image)
    # show_image(GRB_image, name='GRB_iron_man', save=True)
    #
    # #2.1
    # barbara = load_image('barbara.jpg')
    # barbara_gray = convert2gray(barbara)
    # show_image(barbara)
    # show_image(barbara_gray, gray=True, name='barbara_gray', save=True)
    # #2.2
    # plot_hist(barbara_gray, name='barbara_gray_hist', save=True)
    # #2.3
    # barbara_gray_blur2 = gaussianblur(barbara_gray, 2)
    # barbara_gray_blur8 = gaussianblur(barbara_gray, 8)
    # show_image(barbara_gray_blur2, gray=True, name='barbara_gray_blur2', save=True)
    # show_image(barbara_gray_blur8, gray=True, name='barbara_gray_blur8', save=True)
    # #2.4
    # plot_hist(barbara_gray_blur2, name='barbara_gray_blur2_hist', save=True)
    # plot_hist(barbara_gray_blur8, name='barbara_gray_blur8_hist', save=True)
    # #2.5
    # subtract = cv2.subtract(barbara_gray, barbara_gray_blur2)
    # threshold = np.max(np.max(subtract, axis=0)) * 0.05
    # subtract[np.where(subtract < threshold)] = 0
    # show_image(subtract, gray=True, name='subtract', save=True)

    #Filtering
    I1 = np.array([[120, 110, 90, 115, 40],
                   [145, 135, 135, 65, 35],
                   [125, 115, 55, 35, 25],
                   [80, 45, 45, 20, 15],
                   [40, 35, 25, 10, 10]])
    I2 = np.array([[125, 130, 135, 110, 125],
                   [145, 135, 135, 155, 125],
                   [65, 60, 55, 45, 40],
                   [40, 35, 40, 25, 15],
                   [15, 15, 20, 15, 10]])
    filter1 = 1/3 * np.array(np.ones(3)).reshape((3,1))
    filter2 = 1/3 * np.array([[1, 1, 1]])
    filter3 = np.array(1/9 * np.ones((3,3)))
    matrix_filter(I1, filter1)





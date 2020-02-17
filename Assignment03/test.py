import numpy as np
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy.misc import imresize
def gauss2D(shape=(3, 3),sigma=0.5):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def normalize(img):
    ''' Function to normalize an input array to 0-1 '''
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)
def my_imfilter(image, imfilter):
    output = np.zeros_like(image)
    pad_x = (imfilter.shape[0] - 1) // 2
    pad_y = (imfilter.shape[1] - 1) // 2
    for ch in range(image.shape[2]):
        image_pad = np.lib.pad(image[:, :, ch], ((pad_x, pad_x), (pad_y, pad_y)), 'constant', constant_values=(0, 0))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                # multiply first and sum together, i.e. convolution
                output[i, j, ch] = np.sum(
                    np.multiply(image_pad[i:i + imfilter.shape[0], j:j + imfilter.shape[1]], imfilter))
    return output


def vis_hybrid_image(hybrid_image):

    scales = 5  # how many downsampled versions to create
    scale_factor = 0.5  # how much to downsample each time
    padding = 5  # how many pixels to pad.

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]  # counting how many color channels the input has
    output = hybrid_image[:]
    cur_image = hybrid_image[:]

    for i in range(1, scales):
        # add padding
        output = np.concatenate((output, np.ones((original_height, padding, num_colors))), axis=1)

        # dowsample image;
        cur_image = imresize(cur_image, scale_factor, 'bilinear').astype(np.float) / 255
        # pad the top and append to the output
        tmp = np.concatenate(
            (np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)

    return output
def main():
    image1 = mpimg.imread('marilyn.bmp')
    image2 = mpimg.imread('einstein.bmp')
    image1 = image1.astype(np.float32)/255
    image2 = image2.astype(np.float32)/255

    cutoff_frequency = 3
    gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1,cutoff_frequency*4+1), sigma = cutoff_frequency)
    low_frequencies = my_imfilter(image1, gaussian_filter)
    high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
    hybrid_image = low_frequencies + high_frequencies

    plt.figure(1)
    plt.imshow(low_frequencies)
    plt.figure(2)
    plt.imshow(high_frequencies+0.5)
    vis = vis_hybrid_image(hybrid_image)
    plt.figure(3)
    plt.imshow(vis)

    plt.show()




if __name__ == '__main__':
    main()
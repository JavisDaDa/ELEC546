import cv2
from scipy.misc import imresize
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp
'''
cite from https://github.com/howard19950724/Image-Filtering-and-Hybrid-Images
'''
def vis_hybrid_image(hybrid_image):

    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]
    output = hybrid_image[:]
    cur_image = hybrid_image[:]

    for i in range(1, scales):
        output = np.concatenate((output, np.ones((original_height, padding, num_colors))), axis=1)
        cur_image = imresize(cur_image, scale_factor, 'bilinear').astype(np.float) / 255
        tmp = np.concatenate(
            (np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)

    return output
if __name__ == '__main__':
    image1 = utils.load_image('test000.jpg')
    image2 = utils.load_image('test0000.jpg')
    # image1 = utils.load_image('marilyn.bmp')
    # image2 = utils.load_image('einstein.bmp')
    image1 = image1.astype(np.float32) / 255
    image2 = image2.astype(np.float32) / 255
    cutoff_frequency = 15
    low = cv2.GaussianBlur(image1, (cutoff_frequency * 4 + 1, cutoff_frequency * 4 + 1), cutoff_frequency)
    high = image2 - cv2.GaussianBlur(image2, (cutoff_frequency * 4 + 1, cutoff_frequency * 4 + 1), cutoff_frequency)
    hybrid_image = low + high
    vis = vis_hybrid_image(hybrid_image)
    plt.figure(1)
    plt.gray()
    plt.imshow(hybrid_image)
    plt.axis('off')
    plt.savefig('hahaha1.png')
    plt.figure(2)
    plt.gray()
    plt.imshow(vis)
    plt.axis('off')
    plt.savefig('vishahahaha1.png')

    plt.show()

# image1 = utils.load_image('test000.jpg')
# image2 = utils.load_image('test0000.jpg')
# image1 = image1[:1400, 40:1040, :]
# image2 = image2[100:1500, 70:1070, :]
#
# plt.imshow(image1)
# mp.imsave('test000.jpg', image1)
# plt.show()
# plt.imshow(image2)
# mp.imsave('test0000.jpg', image2)
# plt.show()
# print(image1.shape)
# print(image2.shape)
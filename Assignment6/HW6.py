'''
author: Yunda Jia
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
import scipy.io

# https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


def disparity(imgL, imgR):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity, 'gray', extent=[0, 100, 0, 1], aspect='auto')
    plt.savefig('disparity.png')
    plt.show()



def similarity(imgL, imgR, pixel, patchsize):
    m, n = imgL.shape

    top = int(max(pixel[0] - patchsize / 2, 1))
    bottom = int(min(pixel[0] + patchsize / 2, m))
    left = int(max(pixel[1] - patchsize / 2, 1))
    right = int(min(pixel[1] + patchsize / 2, n))

    patch = imgL[top: bottom, left:right]
    xcorr = normxcorr2(patch, imgR[top:bottom][:])
    sim = xcorr[patchsize, patchsize // 2:n + patchsize // 2]
    return sim

def plotsimilarity(p, imgL):
    m, n = imgL.shape
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    for i, pixel in enumerate(p):
        sim = similarity(imgL, imgR, pixel, patchsize=30)
        x = np.arange(0 - pixel[1], n - pixel[1])
        axes[i // 3][i % 3].plot(x, sim)
        axes[i // 3][i % 3].set_title(f'p={str(i + 1)}')
    plt.savefig('similarity.png')
    plt.show()

def dynamic_disparity(imgL, imgR, patchsize):
    m, n = imgL.shape
    dydist = np.zeros((m, n))
    for row in range(m):
        line = np.zeros((1, n))
        dis = np.zeros(n)
        for j in range(n):
            pixel = [row, j]
            dis[j, :] = similarity(imgL, imgR, pixel, patchsize)

        path = np.zeros(n)
        for j in range(1, n):
            path[1, j] = path[1, j - 1] + dis[1, j]
            path[j, 1] = path[j - 1, 1] + dis[1, j]

        for i in range(1, n):
            for j in range(1, n):
                path[i, j] = max(path[i - 1, j], path[i - 1, j - 1], path[i, j - 1]) + dis(i, j)

        col = n
        line = np.zeros(col)
        j = n
        while j > 1:
            if col == 1:
                line[j - 1] = j - 1
                j -= 1
            else:
                temp = max(path[j, col - 1], path[j - 1, col], path[j - 1, col - 1])
                if temp == path[j, col - 1]:
                    col -= 1
                elif temp == path[j - 1, col]:
                    line[j - 1] = j - col + 1
                    j -= 1
                else:
                    line[j - 1] = j - col
                    j -= 1
                    col -= 1
        dydist[row, :] = line
    return dydist


if __name__ == '__main__':
    imgL = cv2.imread('./BinocularStereo/tsukuba_l.ppm', 0)
    imgR = cv2.imread('./BinocularStereo/tsukuba_r.ppm', 0)
    p = [[136, 83], [203, 304], [182, 119], [186, 160], [123, 224], [153, 338]]
    # problem1.1.1 & problem1.1.2
    plotsimilarity(p, imgL)
    # problem 1.1.3 & 1.1.3
    disparity(imgL, imgR)
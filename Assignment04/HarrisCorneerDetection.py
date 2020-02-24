import utils
import numpy as np
from numpy.fft import fft2, ifft2
import cv2
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
class HarrisCornerDetector():

    def __init__(self):
        pass

    def my_filter(self, f, im):
        '''
        Define a filter using FFT and inverse FFT and use it to an image.
        :param f: filter
        :param im: image
        :return: filtered image
        '''
        image = im
        kernel = np.zeros(image.shape)
        kernel[:f.shape[0], :f.shape[1]] = f
        kernel = fft2(kernel)
        fim = fft2(image)
        res = np.real(ifft2(kernel * fim)).astype(float)
        return res

    def cornerness(self, A, B, C, epsilon=0.01):
        detH = A * C - np.square(B)
        traceH = A + C
        M = detH / (epsilon + traceH)
        return M

    def NMS(self, M, distance, threshold):
        res = peak_local_max(M, distance, threshold)
        return res

    def visulize(self, img_toshow, res, radius, color, thickness):
        for i, j in res:
            cv2.circle(img_toshow, (j, i), radius, color, thickness)
        plt.imshow(img_toshow)
        plt.axis('off')
        plt.savefig('resize.png')
        plt.show()
        pass


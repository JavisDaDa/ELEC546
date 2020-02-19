import utils
import numpy as np
from numpy.fft import fft2, ifft2
import cv2

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


import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import scipy.ndimage.filters
import scipy.signal
from numpy.fft import fft2, ifft2
class Assignment3(object):
    def __init__(self):
        pass


    def gaussian(self, im):
        '''
        :param im:
        :return:
        '''
        image = im
        b = np.array([2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2]).reshape((5, 5)) / 159
        kernel = np.zeros(image.shape)
        kernel[:b.shape[0], :b.shape[1]] = b
        fim = fft2(image)
        fkernel = fft2(kernel)
        fil_im = ifft2(fim * fkernel)
        return abs(fil_im).astype(int)

    def my_filter(self, f, im):
        '''
        :param f:
        :param im:
        :return:
        '''
        image = im
        kernel = np.zeros(image.shape)
        kernel[:f.shape[0], :f.shape[1]] = f
        kernel = fft2(kernel)
        fim = fft2(image)
        res = np.real(ifft2(kernel * fim)).astype(float)
        return res

    def gradient(self, Dx, Dy):
        '''
        :param Dx:
        :param Dy:
        :return:
        '''
        D = np.sqrt(np.square(Dx) + np.square(Dy))
        theta = np.arctan2(Dy, Dx) * 180 / np.pi
        theta[np.where(theta < 0)] += 360
        theta1 = np.zeros(Dx.shape)
        theta1[(np.where(theta >= 0) and np.where(theta < 22.5)) or (np.where(theta >= 337.5) and np.where(theta <= 360)) or (np.where(theta >= 157.5) and np.where(theta <= 202.5))] = 0
        theta1[(np.where(theta >= 22.5) and np.where(theta < 67.5)) or (np.where(theta >= 202.5) and np.where(theta < 247.5))] = 45
        theta1[(np.where(theta >= 67.5) and np.where(theta < 112.5)) or (np.where(theta >= 247.5) and np.where(theta < 292.5))] = 90
        theta1[(np.where(theta >= 112.5) and np.where(theta < 157.5)) or (np.where(theta >= 292.5) and np.where(theta < 337.5))] = 135
        return D, theta1

    def non_maximum_suppression(self, gradient, theta):
        '''
        :param gradient:
        :param theta:
        :return:
        '''
        image = np.zeros(gradient.shape)
        m, n = image.shape
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if theta[i][j] == 0:
                    if gradient[i][j] >= gradient[i + 1][j] and gradient[i][j] >= gradient[i - 1][j]:
                        image[i][j] = gradient[i][j]
                    else:
                        continue
                elif theta[i][j] == 45:
                    if gradient[i][j] >= gradient[i + 1][j + 1] and gradient[i][j] >= gradient[i - 1][j - 1]:
                        image[i][j] = gradient[i][j]
                    else:
                        continue
                elif theta[i][j] == 90:
                    if gradient[i][j] >= gradient[i][j + 1] and gradient[i][j] >= gradient[i][j - 1]:
                        image[i][j] = gradient[i][j]
                    else:
                        continue
                elif theta[i][j] == 135:
                    if gradient[i][j] >= gradient[i - 1][j + 1] and gradient[i][j] >= gradient[i + 1][j - 1]:
                        image[i][j] = gradient[i][j]
                    else:
                        continue
                else:
                    continue
        return image



if __name__ == '__main__':
    # path = 'cameraman.tif'
    path = 'emilia.jpg'
    HW3 = Assignment3()
    image = utils.load_image(path)[:, :, 0]
    # 2.a

    fim = HW3.gaussian(image)

    # 2.b.1
    fx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, -1))
    Dx = HW3.my_filter(fx, fim)
    fy = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3, -1))
    Dy = HW3.my_filter(fy, fim)

    # 2.b.2
    D, theta = HW3.gradient(Dx, Dy)
    # 2.c

    nms_image = HW3.non_maximum_suppression(D, theta)
    utils.show_image(nms_image, gray=True)
    print('\n')

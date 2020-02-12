import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import scipy.ndimage.filters
import scipy.signal
from numpy.fft import fft2, ifft2
class Assignment3(object):
    def __init__(self, path):
        self.path = path
        # self.image = utils.convert2gray(utils.load_image(self.path))
        self.image = utils.load_image(self.path)[:, :, 0]

    def gaussian(self, f):
        # res = cv2.filter2D(self.image, -1, f)
        # res = scipy.ndimage.convolve(self.image, f, mode='nearest')
        kernel = np.zeros(self.image.shape)
        kernel[:f.shape[0], :f.shape[1]] = f
        fim = fft2(self.image)
        fkernel = fft2(kernel)
        fil_im = ifft2(fim * fkernel)
        # plt.figure()
        # plt.subplot(121)
        # plt.title('Original Image')
        # plt.imshow(self.image[:, :-1], cmap=plt.cm.gray)
        # plt.subplot(122)
        # plt.title('filtered image')
        # plt.imshow(res[:, :-1],  cmap=plt.cm.gray)
        # if save:
        #     plt.savefig(f'{name}.png')
        # plt.show()
        # return res
        return abs(fil_im).astype(int)

    def my_filter(self, f, name, save=False):
        kernel = np.zeros(self.image.shape)
        kernel[:f.shape[0], :f.shape[1]] = f
        kernel = fft2(kernel)
        fim = fft2(self.image)
        res = np.real(ifft2(kernel * fim)).astype(float)
        return res

    def gradient(self, Dx, Dy):
        D = np.sqrt(np.square(Dx) + np.square(Dy))
        test = Dy / Dx
        test2 = np.arctan(test)
        theta = np.arctan(Dy / Dx) * 180 / np.pi
        # theta[np.where(theta < 0)] += 360
        theta1 = np.zeros((256, 256))
        m, n = theta.shape
        for i in range(m):
            for j in range(n):
                if theta[i][j] < 0:
                    theta[i][j] += 360
                degree = theta[i][j]
                if (0 <= degree < 22.5) or (337.5 <= degree <= 360) or (157.5 <= degree < 202.5):
                    theta1[i][j] = 0
                elif (22.5 <= degree < 67.5) or (202.5 <= degree < 247.5):
                    theta1[i][j] = 45
                elif (67.5 <= degree < 112.5) or (247.5 <= degree < 292.5):
                    theta1[i][j] = 90
                elif (112.5 <= degree < 157.5) or (292.5 <= degree < 337.5):
                    theta1[i][j] = 135
        # theta1[(np.where(theta >= 0) and np.where(theta < 22.5)) or (np.where(theta >= 337.5) and np.where(theta <= 360)) or (np.where(theta >= 157.5) and np.where(theta <= 202.5))] = 0
        # theta1[(np.where(theta >= 22.5) and np.where(theta < 67.5)) or (np.where(theta >= 202.5) and np.where(theta < 247.5))] = 45
        # theta1[(np.where(theta >= 67.5) and np.where(theta < 112.5)) or (np.where(theta >= 247.5) and np.where(theta < 292.5))] = 90
        # theta1[(np.where(theta >= 112.5) and np.where(theta < 157.5)) or (np.where(theta >= 292.5) and np.where(theta < 337.5))] = 135
        # theta1[np.where(tantheta < np.tan(np.pi / 8)) and np.where(tantheta > -np.tan(np.pi / 8))] = 0
        # theta1[np.where(tantheta == float('inf'))] = np.pi / 2
        # theta1[np.where(tantheta >= np.tan(np.pi / 8)) and np.where(tantheta < np.tan(3 * np.pi / 8))] = np.pi / 4
        # theta1[np.where(tantheta >= np.tan(3 * np.pi / 8)) and np.where(tantheta < -np.tan(3 * np.pi / 8))] = np.pi / 2
        # theta1[np.where(tantheta <= -np.tan(3 * np.pi / 8)) and np.where(tantheta <= -np.tan(np.pi / 8))] = 3 * np.pi / 4
        return D, theta1

    def non_maximum_suppression(self, im, theta, gradient):
        image = np.zeros(im.shape)
        m, n = image.shape
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if theta[i][j] == 0:
                    if gradient[i][j] >= gradient[i + 1][j] and gradient[i][j] >= gradient[i - 1][j]:
                        image[i][j] = 255
                    else:
                        image[i][j] = 0
                elif theta[i][j] == 45:
                    if gradient[i][j] >= gradient[i + 1][j + 1] and gradient[i][j] >= gradient[i - 1][j - 1]:
                        image[i][j] = 255
                    else:
                        image[i][j] = 0
                elif theta[i][j] == 90:
                    if gradient[i][j] >= gradient[i][j + 1] and gradient[i][j] >= gradient[i][j - 1]:
                        image[i][j] = 255
                    else:
                        image[i][j] = 0
                elif theta[i][j] == 135:
                    if gradient[i][j] >= gradient[i - 1][j + 1] and gradient[i][j] >= gradient[i + 1][j - 1]:
                        image[i][j] = 255
                    else:
                        image[i][j] = 0
                else:
                    continue
        return image




if __name__ == '__main__':
    path = 'cameraman.tif'
    HW3 = Assignment3(path)
    # 2.a
    f = np.array([2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2]).reshape((5, 5)) / 159
    res = HW3.gaussian(f)
    res -= 2

    # 2.b.1
    fx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, -1))
    Dx = HW3.my_filter(fx, 'Dx', save=True)
    fy = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3, -1))
    Dy = HW3.my_filter(fy, 'Dy', save=True)

    # 2.b.2
    D, theta = HW3.gradient(Dx, Dy)
    utils.show_image(theta)
    # the = Dy / Dx
    # theta = np.arctan(Dy / Dx) * 360 / (2 * np.pi)
    # 2.c
    nms_image = HW3.non_maximum_suppression(theta, res, D)
    utils.show_image(nms_image, gray=True)
    print('\n')

import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt


class CannyEdgeDetection(object):
    def __init__(self):
        pass

    def gaussian(self, im):
        '''
        Define a gaussian filter.
        :param im: image to be filtered
        :return: filtered image
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

    def gradient(self, Dx, Dy):
        '''
        Derive the total gradient of an image.
        :param Dx: horizontal gradient
        :param Dy: vertical gradient
        :return: total gradient and degrees of each pixel
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
        Define non-maximum suppression.
        :param gradient: total gradient of an image.
        :param theta: degrees of gradients in an image.
        :return: filtered image
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

    def Hysteresis(self,img, D, t_low, t_high):
        '''
        Define Hysteresis threshold
        :param img: image
        :param D: total gradient
        :param t_low: low threshold
        :param t_high: high threshold
        :return: image
        '''
        direction3 = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
        direction5 = [(-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
                      (-2, -1), (2, -1),
                      (-2, 0), (2, 0),
                      (-2, 1), (2, 1),
                      (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2)]
        m, n = img.shape
        for i in range(m):
            for j in range(n):
                if img[i][j] == D[i][j]:
                    if D[i][j] < t_low:
                        img[i][j] = 0
                    elif D[i][j] > t_high:
                        img[i][j] = 255
                        continue
                    elif t_low <= D[i][j] <= t_high:
                        flag = False
                        for dx, dy in direction3:
                            x = i + dx
                            y = j + dy
                            if 0 <= x < m and 0 <= y < n and D[x][y] > t_high:
                                flag = True
                                img[i][j] = 255
                                break
                            else:
                                continue
                        if not flag:
                            for dx, dy in direction3:
                                x = i + dx
                                y = j + dy
                                if 0 <= x < m and 0 <= y < n and t_low <= D[x][y] <= t_high:
                                    flag = True
                                    break
                                else:
                                    continue
                        if flag:
                            for dx, dy in direction5:
                                x = i + dx
                                y = j + dy
                                if 0 <= x < m and 0 <= y < n and D[x][y] > t_high:
                                    flag = True
                                    img[i][j] = 255
                                    break
                                else:
                                    continue
                        else:
                            img[i][j] = 0
        return img

    def show_images(self, images, name, save=False, row=2):
        '''
        :param images: images to show
        :param name: names to title
        :param save: save or not
        :param row: how many rows
        :return: None
        '''
        m = len(images)
        for i, image in enumerate(images):
            plt.subplot(row, m // row, i + 1)
            plt.imshow(image)
            plt.title(f'{name[i]}')
            plt.axis('off')
            plt.gray()
        if save:
            plt.savefig(f'{name[-1]}.png')
        plt.show()


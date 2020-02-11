import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import scipy.ndimage.filters
class Assignment3(object):
    def __init__(self, path):
        self.path = path
        self.image = utils.load_image(self.path)

    def my_filter(self, f, name, save=False):
        res = cv2.filter2D(self.image, -1, f)
        plt.figure()
        plt.subplot(121)
        plt.title('Original Image')
        plt.imshow(self.image[:, :, ::-1])
        plt.subplot(122)
        plt.title('sharpen_1 Image')
        plt.imshow(res[:, :, ::-1])
        if save:
            plt.savefig(f'{name}.png')
        plt.show()
        return res
if __name__ == '__main__':
    path = 'cameraman.tif'
    HW3 = Assignment3(path)
    # 2.a
    f = np.array([2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2]).reshape((5, 5)) / 159
    # HW3.my_filter(f, 'noisereduction', save=True)

    # 2.b.1
    fx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, -1))
    Dx = HW3.my_filter(fx, 'Dx', save=False)
    fy = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3, -1))
    Dy = HW3.my_filter(fy, 'Dy', save=False)

    # 2.b.2
    D = np.sqrt(np.square(Dx) + np.square(Dy))

import cv2
import numpy as np
import utils
from CannyEdgeDetection import CannyEdgeDetection
import matplotlib.pyplot as plt

def main(path):
    '''
    Main function
    :param path: image path
    :return: None
    '''
    HW3 = CannyEdgeDetection()
    image = utils.load_image(path)[:, :, 0]
    # 2.a
    fim = HW3.gaussian(image)
    plt.imshow(fim)
    plt.gray()
    plt.axis('off')
    plt.savefig(f'2a.png')
    # 2.b.1
    fx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, -1))
    Dx = HW3.my_filter(fx, fim)
    plt.imshow(Dx)
    plt.gray()
    plt.axis('off')
    plt.savefig(f'2b1.png')
    fy = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3, -1))
    Dy = HW3.my_filter(fy, fim)
    plt.imshow(Dy)
    plt.gray()
    plt.axis('off')
    plt.savefig(f'2b12.png')

    # 2.b.2
    D, theta = HW3.gradient(Dx, Dy)
    # 2.c
    nms_image = HW3.non_maximum_suppression(D, theta)
    plt.imshow(nms_image)
    plt.gray()
    plt.axis('off')
    plt.savefig(f'2c.png')
    # 2.d
    edge = HW3.Hysteresis(nms_image, D, 30, 100)
    plt.imshow(edge)
    plt.gray()
    plt.axis('off')
    plt.savefig(f'2d.png')
    images = [image, fim, Dx, Dy, nms_image, edge]
    names = ['original', 'Noise Reduction', 'Dx', 'Dy', 'Non-maximum suppression', 'final', 'Canny edge detection']
    HW3.show_images(images, names, save=True)
    # 2.e
    canny = cv2.Canny(image, 30, 100)
    images2 = [edge, canny]
    names2 = ['My canny', 'canny library', 'Comparision']
    HW3.show_images(images2, names2, save=True, row=1)


if __name__ == '__main__':
    main('cameraman.tif')


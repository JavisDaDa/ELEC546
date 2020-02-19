import utils
import HarrisCorneerDetection
import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    HW4 = HarrisCorneerDetection.HarrisCornerDetector()
    # 1.a
    img = utils.convert2gray(utils.load_image('chessboard.jpg'))
    img_toshow = utils.load_image('chessboard.jpg')
    utils.show_image(img, gray=True)
    # 1.b
    fx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, -1))
    Ix = HW4.my_filter(fx, img)
    plt.imshow(Ix)
    plt.gray()
    plt.axis('off')
    plt.show()
    # plt.savefig(f'2b1.png')
    fy = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3, -1))
    Iy = HW4.my_filter(fy, img)
    plt.imshow(Iy)
    plt.gray()
    plt.axis('off')
    # plt.savefig(f'2b12.png')
    plt.show()
    # 1.c
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = Ix * Iy
    # 1.d, 1.e
    GIx2 = utils.gaussianblur(Ix2, 2)
    GIy2 = utils.gaussianblur(Iy2, 2)
    GIxy = utils.gaussianblur(Ixy, 2)
    # 1.f
    epsilon = 0.01
    M = HW4.cornerness(GIx2, GIxy, GIy2, epsilon)
    # 2.a
    threshold = 10000
    distance = 3
    res = HW4.NMS(M, distance, threshold)
    # 2.b, 2.c
    radius = 1
    color = (0, 255, 0)  # Green
    thickness = 3
    HW4.visulize(img_toshow, res, radius, color, thickness)


if __name__ == '__main__':
    main()
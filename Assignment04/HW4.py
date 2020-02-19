import utils
import HarrisCorneerDetection
import numpy as np
import matplotlib.pyplot as plt

def main():
    HW4 = HarrisCorneerDetection.HarrisCornerDetector()
    # a
    img = utils.convert2gray(utils.load_image('chessboard.jpg'))
    utils.show_image(img, gray=True)
    # b
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
    # c
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = Ix * Iy
    # d, e
    GIx2 = utils.gaussianblur(Ix2, 4)
    GIy2 = utils.gaussianblur(Iy2, 4)
    GIxy = utils.gaussianblur(Ixy, 4)


if __name__ == '__main__':
    main()
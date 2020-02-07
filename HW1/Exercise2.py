import utils
import cv2
import numpy as np
if __name__ == '__main__':
    #2.1
    barbara = utils.load_image('barbara.jpg')
    barbara_gray = utils.convert2gray(barbara)
    utils.show_image(barbara)
    utils.show_image(barbara_gray, gray=True, name='barbara_gray', save=True)
    #2.2
    utils.plot_hist(barbara_gray, name='barbara_gray_hist', save=True)
    #2.3
    barbara_gray_blur2 = utils.gaussianblur1(barbara_gray, 2)
    barbara_gray_blur8 = utils.gaussianblur1(barbara_gray, 8)
    utils.show_image(barbara_gray_blur2, gray=True, name='barbara_gray_blur2', save=True)
    utils.show_image(barbara_gray_blur8, gray=True, name='barbara_gray_blur8', save=True)
    #2.4
    utils.plot_hist(barbara_gray_blur2, name='barbara_gray_blur2_hist', save=True)
    utils.plot_hist(barbara_gray_blur8, name='barbara_gray_blur8_hist', save=True)
    #2.5
    subtract = cv2.subtract(barbara_gray, barbara_gray_blur2)
    threshold = np.max(np.max(subtract, axis=0)) * 0.05
    subtract[np.where(subtract < threshold)] = 0
    utils.show_image(subtract, gray=True, name='subtract', save=True)

    # See Exercise.m for the implementation of applying filters

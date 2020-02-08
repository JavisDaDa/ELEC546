import utils
import cv2
import numpy as np
if __name__ == '__main__':
    #2.1
    barbara = utils.load_image('barbara.jpg')
    barbara_gray = utils.convert2gray(barbara)
    utils.show_image(barbara)
    utils.show_image(barbara_gray, gray=True, name='barbaragray', save=True)
    #2.2
    utils.plot_hist(barbara_gray, name='barbaragrayhist', save=True)
    #2.3
    barbara_gray_blur2 = utils.gaussianblur1(barbara_gray, 2)
    barbara_gray_blur8 = utils.gaussianblur1(barbara_gray, 8)
    utils.show_image(barbara_gray_blur2, gray=True, name='barbaragrayblur2', save=True)
    utils.show_image(barbara_gray_blur8, gray=True, name='barbaragrayblur8', save=True)
    #2.4
    utils.plot_hist(barbara_gray_blur2, name='barbaragrayblur2hist', save=True)
    utils.plot_hist(barbara_gray_blur8, name='barbaragrayblur8hist', save=True)
    #2.5
    subtract = cv2.subtract(barbara_gray, barbara_gray_blur2)
    threshold = np.max(np.max(subtract, axis=0)) * 0.05
    subtract[np.where(subtract < threshold)] = 0
    utils.show_image(subtract, gray=True, name='subtract', save=True)

    # See Exercise.m for the implementation of applying filters

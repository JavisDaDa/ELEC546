import utils
import cv2
if __name__ == '__main__':
    # Edge preserving smoothing
    camera_man_noisy = utils.load_image('camera_man_noisy.png')
    utils.show_image(camera_man_noisy)
    res = cv2.bilateralFilter(camera_man_noisy, 9, 75, 75)
    utils.show_image(res)
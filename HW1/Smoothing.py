import utils
if __name__ == '__main__':
    # smoothing
    camera_man_noisy = utils.load_image('camera_man_noisy.png')
    utils.show_image(camera_man_noisy)
    for size in ((2, 2), (4, 4), (8, 8), (16, 16)):
        test = utils.average_filter(camera_man_noisy, size)
        utils.show_image(test)
    for std in [2, 4, 8, 16]:
        test = utils.gaussianblur(camera_man_noisy, std)
        utils.show_image(test)
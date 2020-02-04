import utils
if __name__ == '__main__':
    path = 'Iron_man.jpg'
    image = utils.load_image(path)
    utils.show_image(image)
    # 1.1
    image_head = utils.crop_head(image)
    # 1.2
    utils.show_image(image_head, name='iron_man_head', save=True)
    # 1.3
    image_green = utils.RGBcomponent(image_head, RGB='G')
    utils.show_image(image_green, name='iron_man_head_green', save=True)
    # 1.4
    GRB_image = utils.RGB2GRB(image)
    utils.show_image(GRB_image, name='GRB_iron_man', save=True)

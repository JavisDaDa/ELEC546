import utils
import imageio
if __name__ == '__main__':
    path = 'Iron_man.jpg'
    image = utils.load_image(path)
    utils.show_image(image)
    # 1.1
    image_head = utils.crop_head(image)
    # 1.2
    utils.show_image(image_head, name='ironmanhead', save=True)
    # 1.3
    image_green = utils.RGBcomponent(image_head, RGB='G')
    utils.show_image(image_green, name='ironmanheadgreen', save=True)
    # 1.4
    GRB_image = utils.RGB2GRB(image)
    imageio.imsave('img.png', GRB_image)
    utils.show_image(GRB_image, name='GRBironman', save=True)

from scipy import misc, ndimage
import utils
import numpy as np
import matplotlib.pyplot as plt

def rotate_frame(frame, angle):

    # Perform the image rotation and update the fits header
    #frame[np.isnan(frame)] = 0.0
    new_frame = ndimage.interpolation.rotate(frame, angle, reshape=False, order=1, mode='constant', cval=float('nan'))

    #new_frame = misc.imrotate(frame, angle, interp="bilinear")

    # Return the rotated frame
    return new_frame

img = utils.load_image('chessboard.jpg')
img_rotate = rotate_frame(img, 30)
plt.imshow(img_rotate)
plt.show()
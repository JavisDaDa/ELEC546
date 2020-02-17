import cv2

import utils
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    image1 = utils.load_image('dog.bmp')
    image2 = utils.load_image('cat.bmp')
    image1 = image1.astype(np.float32) / 255
    image2 = image2.astype(np.float32) / 255
    cutoff_frequency = 7
    low = cv2.GaussianBlur(image1, (cutoff_frequency * 4 + 1, cutoff_frequency * 4 + 1), cutoff_frequency)
    high = image2 - cv2.GaussianBlur(image2, (cutoff_frequency * 4 + 1, cutoff_frequency * 4 + 1), cutoff_frequency)
    hybrid_image = low + high
    plt.imshow(hybrid_image)
    plt.axis('off')
    plt.savefig('result.png')
    plt.show()
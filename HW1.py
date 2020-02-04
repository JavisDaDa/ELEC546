import imageio
import matplotlib.pyplot as plt

ironman = imageio.imread('Iron_man.jpg')
print(ironman.shape)
ironman_head = ironman[200:450, 260:450, :]
plt.imshow(ironman)
plt.imshow(ironman_head)
plt.savefig('ironman_head.png')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
filename = 'chessboard.jpg'
img = cv2.imread(filename)
img_toshow = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img_toshow[dst>0.01*dst.max()]=[0,255,0]

plt.imshow(img_toshow)
plt.show()
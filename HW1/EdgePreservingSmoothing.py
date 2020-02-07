# import utils
# import cv2
# import numpy as np
# if __name__ == '__main__':
#     # Edge preserving smoothing
#     camera_man_noisy = cv2.imread('camera_man_noisy.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
#     B = np.stack([
#         utils.filter_bilateral(camera_man_noisy[:, :, 0], 10.0, 0.1),
#         utils.filter_bilateral(camera_man_noisy[:, :, 1], 10.0, 0.1),
#         utils.filter_bilateral(camera_man_noisy[:, :, 2], 10.0, 0.1)], axis=2)
#     O = np.hstack([camera_man_noisy, B])
#     #cv2.imwrite('images/lena_bilateral.png', out * 255.0)
#     #camera_man_noisy = utils.convert2gray(camera_man_noisy)
#     # camera_man_noisy[camera_man_noisy < 0] = 0
#     # camera_man_noisy[camera_man_noisy > 1] = 1
#     utils.show_image(camera_man_noisy, gray=True)
#     utils.show_image(O, gray=True)
#     # utils.show_image(camera_man_noisy, gray=True)
#     # res = cv2.bilateralFilter(camera_man_noisy, 1, 77, 77)
#     # utils.show_image(res, gray=True)
import cv2 as cv
# import numpy as np
# import utils
# def nothing(x):
#     pass
# cv.namedWindow("image")
# cv.createTrackbar("d","image",0,255,nothing)
# cv.createTrackbar("sigmaColor","image",0,255,nothing)
# cv.createTrackbar("sigmaSpace","image",0,255,nothing)
# img = cv.imread("camera_man_noisy.png",0)
# #img = utils.load_image('camera_man_noisy.png')
# while(1):
#     d = cv.getTrackbarPos("d","image")
#     sigmaColor = cv.getTrackbarPos("sigmaColor","image")
#     sigmaSpace = cv.getTrackbarPos("sigmaSpace","image")
#     out_img = cv.bilateralFilter(img,d,sigmaColor,sigmaSpace)
#     cv.imshow("out",out_img)
#     k = cv.waitKey(1) & 0xFF
#     if k ==27:
#         break
# cv.destroyAllWindows()
print(cv.__version__)
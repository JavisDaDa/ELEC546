import cv2 as cv
import utils

def nothing(x):
    print(x)
cv.namedWindow("image")
cv.createTrackbar("d","image",0,10,nothing)
cv.createTrackbar("sigmaColor","image",0,255,nothing)
cv.createTrackbar("sigmaSpace","image",0,255,nothing)
img = cv.imread("camera_man_noisy.png",0)
while(1):
    d = cv.getTrackbarPos("d","image")
    sigmaColor = cv.getTrackbarPos("sigmaColor","image")
    sigmaSpace = cv.getTrackbarPos("sigmaSpace","image")
    out_img = cv.bilateralFilter(img,d,sigmaColor,sigmaSpace)
    cv.imshow("out",out_img)
    # get keyboard input
    k = cv.waitKey(1) & 0xFF
    # if ESC, then quit
    if k == 27:
        break
    # if 's', then show and save image
    elif k == 115:
        name = f'({d},{sigmaColor},{sigmaSpace})'
        utils.show_image(out_img,name=name, gray=True, save=True)
cv.destroyAllWindows()


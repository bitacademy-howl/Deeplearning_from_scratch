import numpy as np
import cv2

def showImage():
    imgfile = 'images/0.jpg'

    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    # cv2.namedWindow('cat', cv2.WINDOW_NORMAL)
    cv2.namedWindow('cat', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('cat', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(img[[[0]]])

showImage()
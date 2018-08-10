import numpy as np
import cv2

def onMouse(x):
    pass

def imgBlending(imgfile1, imgfile2, event):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    window = cv2.namedWindow("ImgPane")
    cv2.createTrackbar("MIXING", 'ImgPane', 0, 100, onMouse)

    mix = cv2.getTrackbarPos("MIXING", 'ImgPane')

    while True:
        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
        frame = cv2.imshow("ImgPane", img)
        # cv2.
        k = cv2.waitKey(1) & 0xFF
        mix = cv2.getTrackbarPos("MIXING", 'ImgPane')

imgBlending("1.jpg", "3.png")




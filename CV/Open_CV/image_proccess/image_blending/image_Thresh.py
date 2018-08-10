import numpy as np
import cv2

img = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)

ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thr2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thr3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thr4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thr5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

ret, thr6 = cv2.threshold(img, 127, 255, cv2.Subdiv2D_PTLOC_ON_EDGE)

# print(ret)

cv2.imshow("original", img)
cv2.imshow("THRESH_BINARY", thr1)
cv2.imshow("THRESH_BINARY_INV", thr2)
cv2.imshow("THRESH_TRUNC", thr3)
cv2.imshow("THRESH_TOZERO", thr4)
cv2.imshow("THRESH_TOZERO_INV", thr5)

cv2.imshow("????", thr6)

cv2.waitKey(0)
cv2.destroyAllWindows()
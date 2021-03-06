import numpy as np
import cv2

def showImage():
    imgfile = 'images/1.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    cv2.imshow('sul', img)

    # 키보드 입력에 대한 처리
    k = cv2.waitKey(0) & 0xFF

    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('c'):
        cv2.imwrite('images/sul_copy.jpg', img)
        cv2.destroyAllWindows()

showImage()

for i in range(100):
    print(i)
    imgfile = 'images/{0}.jpg'.format(i)
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    cv2.imwrite('images/{0}.jpg'.format(i+1), img)

import numpy as np
import cv2
def drawing():
    img = np.zeros((512,512,3), np.unit8)

    # 다양한 색상과 선두께를 가진 도형 그리기

    cv2.line(img, (0,0), (511,511), (255,0,0), 5)
    cv2.rectangle(img, (384,0), (510, 128), (0, 255, 0), 3)
    cv2.circle(img, (477, 63), 63, (0, 0, 255), -1)
    cv2.ellipse(img, (256, 256), (300,50), (100, 50), 0, 0, 100, (255,0,0), -1)

    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(img, "OpenCV", (10,500), font, 4, (255,255,255),2)

    cv2.imshow("drawing", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

drawing()
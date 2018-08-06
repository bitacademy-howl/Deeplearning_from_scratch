from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = Image.open('maxresdefault.jpg')

pix_img = np.array(img)
# R, G, B 3가지 속성을 가지는 ndarray
# 각 원소는 픽셀, 각 로우는 로우, 컬럼은 컬럼.....

print(pix_img.shape)
# 720 * 1280 * 3
# 720 * 1280 픽셀로 이루어진 RGB 이미지...

print(pix_img)


# 그레이 스케일로 처리하는 좁밥같은 거 말고 RGB에서 다루쟈
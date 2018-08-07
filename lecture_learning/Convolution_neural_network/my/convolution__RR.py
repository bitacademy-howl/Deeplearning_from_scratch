from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 이미지 불러오기
img = Image.open('maxresdefault.jpg')

pix_img = np.array(img)
# R, G, B 3가지 속성을 가지는 ndarray
# 각 원소는 픽셀, 각 로우는 로우, 컬럼은 컬럼.....

print(pix_img.shape)
# 720 * 1280 * 3
# 720 * 1280 픽셀로 이루어진 RGB 이미지...
print(pix_img)

weight = tf.constant([[[1,0,0], ]])


plt.imshow(pix_img, cmap='Greys')
plt.show()

# weight = tf.constant([[[1, 10, -1]], [[1, 10, -1]]],
#                       [[[1, 10, -1]], [[1, 10, -1]]], dtype=np.float32)
#
# conv2d = tf.nn.conv2d(pix_img, weight, strides=[1,1,1,1], padding='SAME')
# conv2d_img = conv2d.eval()
# conv2d_img = np.swapaxes(conv2d_img, 0, 3)
#
# for i, one_img in enumerate(conv2d_img):
#     print(one_img.reshape(3, 3))
#     plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3, 3))
# plt.show()
#
# # 그레이 스케일로 처리하는 좁밥같은 거 말고 RGB에서 다루쟈


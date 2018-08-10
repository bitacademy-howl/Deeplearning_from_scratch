import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1], [2], [3]],
                  [[4], [5],[6]],
                  [[7], [8], [9]]]], dtype=np.float32)

weight = tf.constant([[[[1, 10, -1]], [[1, 10, -1]]],
                      [[[1, 10, -1]], [[1, 10, -1]]]], dtype=np.float32)

print("image.shape", image.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
print(conv2d)

conv2d_img = conv2d.eval()

print("conv2D_img.shape : ", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)

print(conv2d_img)

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3, 3), cmap='gray')
plt.show()
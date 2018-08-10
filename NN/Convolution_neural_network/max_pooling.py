import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

image = np.array([[[[4], [3]],
                   [[2], [1]]]], dtype=np.float32)

# max_pool 마스킹 중 가장 큰 값을 select
pool = tf.nn.max_pool(image, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME') # SAME = zero 패팅 with same size

print(pool.shape)
print(pool.eval())

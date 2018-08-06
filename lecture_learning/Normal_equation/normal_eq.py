import numpy as np
# 데이터 로딩 함수
from sklearn.datasets import fetch_california_housing

import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data] # 편향에 대한 입력특성( X0=1)을 추가
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='Y')

XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)

print(X)
print(Y)
print(XT)
print(theta)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)
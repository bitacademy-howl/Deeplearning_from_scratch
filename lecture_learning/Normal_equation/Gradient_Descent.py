import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape

print(housing.data.shape)

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)

#
n_epochs = 1000
learning_rate = 0.01

# np.c_ : Translates slice objects to concatenation along the second axis.
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]
# print(scaled_housing_data_plus_bias, type(scaled_housing_data_plus_bias))

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="Y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
Y_pred = tf.matmul(X, theta, name="predictions")
error = Y_pred - Y

mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
# 4. epoch
#
# 학습용 사진 전체를 딱 한 번 사용했을 때 한 세대(이폭, epoch)이 지나갔다고 말합니다.
# cifar10의 경우 사진 60,000장 중 50,000장이 학습용, 10,000장이 검사용으로 지정되어 있습니다.
# 그런데 max_iter에서 학습에 사진 6,000,000장을 사용하기로 했기 때문에 50,000장의 학습용 사진이
# 여러번 재사용되게 됩니다. 정확히 계산해보면 6,000,000 / 50,000 = 120 이니 한 사진이 120번씩
# 재사용될 것입니다. 이 경우 120 세대(epoch)라고 말합니다. 검사용의 경우 사진 10,000장을
# 사용하기로 했는데 실제로도 사진이 10,000장 있으니 딱 한 세대만 있는 것입니다.
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0 :
            print("에포크", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

print("best_theta : ")
print(best_theta)
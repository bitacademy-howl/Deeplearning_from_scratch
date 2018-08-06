import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist[0])

#
# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# nb_classes = 10
# layer_colsize = 256
#
# X = tf.placeholder(tf.float32, [None, 784])
# Y = tf.placeholder(tf.float32, [None, nb_classes])
#
# W1 = tf.Variable(tf.random_normal([784, layer_colsize]))
# b1 = tf.Variable(tf.random_normal([layer_colsize]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable(tf.random_normal([layer_colsize, layer_colsize]))
# b2 = tf.Variable(tf.random_normal([layer_colsize]))
# L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
#
# W3 = tf.Variable(tf.random_normal([layer_colsize, nb_classes]))
# b3 = tf.Variable(tf.random_normal([nb_classes]))
# hypothesis = tf.matmul(L2, W3) + b3
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

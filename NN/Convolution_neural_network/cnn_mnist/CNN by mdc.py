import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
#
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
#
learning_rate = 0.001
training_epochs = 100
batch_size = 10
#
# def build_CNN_classifier(x):
#     X_img = tf.reshape(X, [-1, 28, 28, 1])
#
#     weight_function_L1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
#     bias_L1 = tf.Variable(tf.constant(0.1, shape=[32]))
#     L1 = tf.nn.conv2d(X_img, weight_function_L1, strides=[1,1,1,1], padding='SAME') + bias_L1
#     activationF_L1 = tf.nn.relu(L1)
#     # Max Pooling 을 이용해서 이미지의 크기를 1/2로 downsampling..
#     L1_pooling = tf.nn.max_pool(activationF_L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#
#     weight_function_L2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
#     L2 = tf.nn.conv2d(L1_pooling, weight_function_L2, strides=[1,1,1,1], padding='SAME')
#     L2 = tf.nn.relu(L2)
#     L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#     L2 = tf.reshape(L2, [-1, 7*7*64])
#
#     W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
#     b = tf.Variable(tf.random_normal([10]))
#     hypothesis = tf.matmul(L2, W3) + b
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
#
# print('Learning started. It takes sometime.')
#
# for epoch in range(training_epochs):
#     avg_cost = 0
#     total_batch = int(mnist.train.num_examples / batch_size)
#     for i in range(total_batch):
#         batch_xs , batch_ys = mnist.train.next_batch(batch_size)
#         feed_dict = {X:batch_xs, Y: batch_ys}
#         c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
#         avg_cost += c / total_batch
#     print("Epoch : ", '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
#
# print('Learning Finished')
#
# correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# print("Accuracy : ", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
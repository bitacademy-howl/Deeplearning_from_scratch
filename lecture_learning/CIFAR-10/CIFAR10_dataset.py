import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets.cifar10 import load_data



def next_batch(num, data, labels):

    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def build_CNN_classifier(x):
    X_img = x

    L1_weight_function = tf.Variable(tf.truncated_normal([5,5,3,64], stddev=5e-2))
    L1_bias = tf.Variable(tf.constant(0.1, shape=[64]))
    L1_h_system_Function = tf.nn.relu(tf.nn.conv2d(X_img, L1_weight_function, strides=[1,1,1,1], padding='SAME') + L1_bias)
    L1_pooling = tf.nn.max_pool(L1_h_system_Function, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    L2_weight_function = tf.Variable(tf.truncated_normal([5,5,64,64], stddev=5e-2))
    L2_bias = tf.Variable(tf.constant(0.1, shape=[64]))
    L2_h_system_Function = tf.nn.relu(tf.nn.conv2d(L1_pooling, L2_weight_function, strides=[1,1,1,1], padding='SAME') + L2_bias)
    L2_pooling = tf.nn.max_pool(L2_h_system_Function, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    L3_weight_function = tf.Variable(tf.truncated_normal([5,5,64,128], stddev=5e-2))
    L3_bias = tf.Variable(tf.constant(0.1, shape=[128]))
    L3_h_system_Function = tf.nn.relu(tf.nn.conv2d(L2_pooling, L3_weight_function, strides=[1,1,1,1], padding='SAME') + L3_bias)

    L4_weight_function = tf.Variable(tf.truncated_normal([5,5,128,128], stddev=5e-2))
    L4_bias = tf.Variable(tf.constant(0.1, shape=[128]))
    L4_h_system_Function = tf.nn.relu(tf.nn.conv2d(L3_h_system_Function, L4_weight_function, strides=[1,1,1,1], padding='SAME') + L4_bias)

    L5_weight_function = tf.Variable(tf.truncated_normal([5,5,128,128], stddev=5e-2))
    L5_bias = tf.Variable(tf.constant(0.1, shape=[128]))
    L5_h_system_Function = tf.nn.relu(tf.nn.conv2d(L4_h_system_Function, L5_weight_function, strides=[1,1,1,1], padding='SAME') + L5_bias)

    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8*8*128, 384], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=384))

    h_conv5_flat = tf.reshape(L5_h_system_Function, [-1, 8*8*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # keep_prob

    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits=logits)

    return y_pred, logits

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)

(x_train, y_train), (x_test, y_test) = load_data()

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

y_pred, logits = build_CNN_classifier(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss=loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = next_batch(128, x_train, y_train_one_hot.eval())
        if batch % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y:batch[1], keep_prob:1.0})
            loss_print = loss.eval(feed_dict={x:batch[0], y:batch[1], keep_prob: 1.0})

            print("반복(Epoch): %d, 트레이닝 데이터 정확도 : %f, 손실 함수(loss) : %f" % (i, train_accuracy, loss_print))
        sess.run(train_step, feed_dict={x:batch[0], y:batch[1], keep_prob: 0.8})

    test_batch = next_batch(10000, x_test, y_train_one_hot.eval())
    print("테스트 데이터 정확도 : %f" % accuracy.eval(feed_dict = {x:test_batch[0], y:test_batch[1], keep_prob:1.0}))
from tensorflow.contrib.eager.python import saver
import tensorflow as tf

w1 = tf.placeholder(tf.float32, name='w1')
w2 = tf.placeholder(tf.float32, name='w2')
b1 = tf.Variable(2.0, dtype=tf.float32, name='bias')
feed_dict = {'w1':4.0, 'w2':8.0}

w3 = w1 + w2
w4 = tf.multiply(w3, b1, name="op_to_restore")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

result = sess.run(w4, {w1:feed_dict['w1'], w2:feed_dict['w2']})
print(result)

saver.save(sess, "./my_test_model", global_step=1000)

########################################################################################################################
# 텐서플로우 모델 읽기

sess = tf.Session()
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))




#
# '''.............'''
# best_theta = ''
# n_epochs = ''
# mse = ''
# training_op = ''
#
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_local_variables())
#
#     for epoch in range(n_epochs):
#         if epoch % 100 == 0:
#             print("에포크", epoch, "MSE = ", mse.eval())
#             save_path = saver.save(sess, "./tmp/my_model.ckpt")
#         sess.run(training_op)
#     best_theta.eval()
#     save_path = saver.save(sess, "./tmp/my_model_final.ckpt")
# '''....'''
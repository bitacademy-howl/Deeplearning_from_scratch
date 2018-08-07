import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

with tf.variable_scope('one_cell') as scope:

    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    print(cell.output_size, cell.state_size)

    # input is one-hot
    x_data = np.array([[[1,0,0,0]]], dtype=np.float32)


    # pp.pprint((x_data))
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())

    # pp.pprint(outputs.eval())

    
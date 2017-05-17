import tensorflow as tf
from functools import partial

xavier = tf.contrib.layers.xavier_initializer

class FoxNet(object):

    def simple_cnn(self, X, y, height, width, n_channels, filter_size, n_filters, n_labels):

        # TODO: Still not totally sure how we arrive at this number, but it doesn't crash
        magic_number = 2457600 #(height * width) / 2 * n_filters

        # Setup variables
        Wconv1 = tf.get_variable("Wconv1", shape=[filter_size, filter_size, n_channels, n_filters])
        bconv1 = tf.get_variable("bconv1", shape=[n_filters])
        W1 = tf.get_variable("W1", shape=[magic_number, n_labels])
        b1 = tf.get_variable("b1", shape=[n_labels])

        # Define our graph
        a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding='SAME') + bconv1
        h1 = tf.nn.relu(a1)
        h1_flat = tf.reshape(h1, [-1, magic_number])
        y_out = tf.matmul(h1_flat, W1) + b1
        return y_out

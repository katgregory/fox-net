import tensorflow as tf
from functools import partial

xavier = tf.contrib.layers.xavier_initializer

class FoxNet(object):

    def simple_cnn(self, X, y):

        X_shape = tf.shape(X)
        height = X_shape[1]
        width = X_shape[2]
        n_channels = X_shape[3]
        n_frames = X_shape[4]
        # TODO Fix variable sizes

        # Setup variables
        Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
        bconv1 = tf.get_variable("bconv1", shape=[32])
        W1 = tf.get_variable("W1", shape=[5408, 10])
        b1 = tf.get_variable("b1", shape=[10])

        # Define our graph
        a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
        h1 = tf.nn.relu(a1)
        h1_flat = tf.reshape(h1,[-1,5408])
        y_out = tf.matmul(h1_flat,W1) + b1
        return y_out

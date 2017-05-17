# Tensorflow model declarations

from foxnet import FoxNet
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tqdm import *

ph = tf.placeholder

class FoxNetModel(object):
    def __init__(self,
                model,
                lr,
                height,
                width,
                channels,
                frames_per_state,
                num_actions,
                verbose = False):

        self.lr = lr
        self.verbose = verbose

        # Placeholders
        # The first dim is None, and gets sets automatically based on batch size fed in
        # count (in train/test set) x 480 (height) x 680 (width) x 3 (channels) x 3 (num frames)
        X = ph(tf.float32, [None, height, width, channels, frames_per_state])
        y = ph(tf.int64, [None])
        is_training = ph(tf.bool)

        foxnet = FoxNet()

        # Build net
        if (model == "simple"):
            y_out = foxnet.simple_cnn(X, y)

        # Define loss
        total_loss = tf.losses.hinge_loss(tf.one_hot(y, num_actions), logits=y_out)
        mean_loss = tf.reduce_mean(total_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(lr) # select optimizer and set learning rate
        train_step = optimizer.minimize(mean_loss)

    def train(self, session, dataset, batch_size):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    def evaluate_prediction(self, session, dataset, batch_size):
        pass

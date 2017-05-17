# Tensorflow model declarations

from foxnet import FoxNet
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tqdm import *
import matplotlib.pyplot as plt

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
            self.probs = foxnet.simple_cnn(X, y)

        # Define loss
        total_loss = tf.losses.hinge_loss(tf.one_hot(y, num_actions), logits=self.probs)
        self.loss = tf.reduce_mean(total_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(lr) # select optimizer and set learning rate
        self.train_step = optimizer.minimize(self.loss)

    #############################
    # TRAINING
    #############################

    def run(self, session, Xd, yd,
            epochs=1, batch_size=20, print_every=100,
            training_now=False, plot_losses=False):

        # TODO: y should go from a list of action keys to indexes in some action array

        # Have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(predict, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)
        
        # Setting up variables we want to compute (and optimizing)
        # If we have a training function, add that to things we compute
        variables = [mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = self.training_step

        # Counter
        iter_cnt = 0
        for e in range(epochs):
            # Keep track of losses and accuracy
            correct = 0
            losses = []

            # Make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # Generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx : start_idx + batch_size]

                # Create a feed dictionary for this batch
                feed_dict = { X: Xd[idx,:],
                              y: yd[idx],
                              is_training: training_now }

                # Get actual batch size
                actual_batch_size = yd[idx].shape[0]

                # Have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                # Aggregate performance stats
                losses.append(loss * actual_batch_size)
                correct += np.sum(corr)

                # Print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
                iter_cnt += 1

            total_correct = correct / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss, total_correct, e+1))

            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()

        return total_loss, total_correct

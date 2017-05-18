# Tensorflow model declarations

from foxnet import FoxNet
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tqdm import *
import matplotlib.pyplot as plt

ph = tf.placeholder

class FoxNetModel(object):

    ############################
    # SET UP GRAPH
    #############################

    def __init__(self,
                model,
                lr,
                height,
                width,
                n_channels,
                multi_frame_state,
                frames_per_state,
                actions,
                cnn_filter_size,
                cnn_n_filters,
                verbose = False):

        self.lr = lr
        self.verbose = verbose
        self.actions = actions

        # Placeholders
        # The first dim is None, and gets sets automatically based on batch size fed in
        # count (in train/test set) x 480 (height) x 680 (width) x 3 (channels) x 3 (num frames)
        if (multi_frame_state):
            self.X = ph(tf.float32, [None, height, width, n_channels, frames_per_state])
        else:
            self.X = ph(tf.float32, [None, height, width, n_channels])
        self.y = ph(tf.int64, [None])
        self.is_training = ph(tf.bool)

        foxnet = FoxNet()

        # Build net
        if (model == "fc"): # Only works if !multi_frame_state
            self.probs = foxnet.fully_connected(self.X, self.y, len(actions))
        elif (model == "simple_cnn"): # Only works if !multi_frame_state
            self.probs = foxnet.simple_cnn(self.X, self.y, cnn_filter_size, cnn_n_filters, len(actions), self.is_training)

        # Define loss
        onehot_labels = tf.one_hot(self.y, len(actions))
        total_loss = tf.losses.softmax_cross_entropy(onehot_labels, logits=self.probs)
        self.loss = tf.reduce_mean(total_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(lr) # Select optimizer and set learning rate
        self.train_step = optimizer.minimize(self.loss)

    #############################
    # RUN GRAPH
    #############################

    def run(self,
            session,
            Xd,
            yd,
            batch_size,
            epochs=1,
            training_now=False,
            validate_incrementally=False,
            X_eval=None,
            y_eval=None,
            print_every=100,
            plot_losses=False,
            plot_accuracies=False):

        # Have tensorflow compute accuracy
        # TODO BUG: When using batches, seems to compare arrs of size (batch_size,) and (total_size,)
        correct_prediction = tf.equal(tf.argmax(self.probs, 1), yd)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        batch_size = yd.shape[0] # TODO: Add back batch sizes

        # Shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        # Setting up variables we want to compute (and optimizing)
        # If we have a training function, add that to things we compute
        variables = [self.loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = self.train_step

        if validate_incrementally:
            correct_validation = tf.equal(tf.argmax(self.probs, 1), y_eval)
            validate_variables = [self.loss, correct_validation]
            validate_losses = []
            validate_accuracies = []

        iter_cnt = 0 # Counter for printing
        epoch_losses = []
        epoch_accuracies = []
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
                feed_dict = { self.X: Xd[idx,:],
                              self.y: yd[idx],
                              self.is_training: training_now }

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
                          .format(iter_cnt, loss, np.sum(corr) * 1.0 / actual_batch_size))
                iter_cnt += 1

            total_correct = correct * 1.0 / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            epoch_losses.append(total_loss)
            epoch_accuracies.append(total_correct)
            print("Epoch {2}, Overall training loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss, total_correct, e+1))

            if (validate_incrementally):
                validate_feed_dict = {
                    self.X: X_eval,
                    self.y: y_eval,
                    self.is_training: False
                }
                loss, corr = session.run(validate_variables, feed_dict=validate_feed_dict)
                validate_losses.append(loss)
                validate_correct = np.sum(corr * 1.0 / X_eval.shape[0])
                validate_accuracies.append(validate_correct)
                print("Epoch {2}, Validation loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(loss, validate_correct, e+1))

        if plot_losses:
            train_line = plt.plot(epoch_losses, label="Training loss")
            if validate_incrementally:
                validate_line = plt.plot(validate_losses, label="Validation loss")
            plt.legend()
            plt.grid(True)
            plt.title('Loss'.format(e+1))
            plt.xlabel('epoch number')
            plt.ylabel('epoch loss')
            plt.show()

        if plot_accuracies:
            train_line = plt.plot(epoch_accuracies, label="Training accuracy")
            if validate_incrementally:
                validate_line = plt.plot(validate_accuracies, label="Validation accuracy")
            plt.legend()
            plt.grid(True)
            plt.title('Accuracy'.format(e+1))
            plt.xlabel('epoch number')
            plt.ylabel('epoch accuracy')
            plt.show()

        return total_loss, total_correct

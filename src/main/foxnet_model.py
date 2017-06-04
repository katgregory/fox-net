# Tensorflow model declarations

import datetime
from foxnet import FoxNet
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tqdm import *
import matplotlib.pyplot as plt
from data_manager import DataManager

ph = tf.placeholder


class FoxNetModel(object):

    ############################
    # SET UP GRAPH
    #############################

    def __init__(self,
                model,
                q_learning,
                lr,
                height,
                width,
                n_channels,
                multi_frame_state,
                frames_per_state,
                available_actions,
                cnn_filter_size,
                cnn_n_filters,
                verbose = False):

        self.lr = lr
        self.verbose = verbose
        self.available_actions = available_actions
        self.num_actions = len(self.available_actions)

        # Placeholders
        # The first dim is None, and gets sets automatically based on batch size fed in
        # count (in train/test set) x 480 (height) x 680 (width) x 3 (channels) x 3 (num frames)
        if multi_frame_state:
            self.X = ph(tf.float32, [None, frames_per_state, height, width, n_channels])
        else:
            self.X = ph(tf.float32, [None, height, width, n_channels])
        self.y = ph(tf.int64, [None])
        self.is_training = ph(tf.bool)

        foxnet = FoxNet()

        # Build net
        if model == "fc": # Only works if !multi_frame_state
            self.probs = foxnet.fully_connected(self.X, self.y, self.num_actions)
        elif model == "simple_cnn": # Only works if !multi_frame_state
            self.probs = foxnet.simple_cnn(self.X, self.y, cnn_filter_size, cnn_n_filters, self.num_actions, multi_frame_state, frames_per_state, self.is_training)
        elif model == "dqn":
            self.probs = foxnet.DQN(self.X, self.y, self.num_actions)
        else:
            raise ValueError("Invalid model specified. Valid options are: 'fcc', 'simple_cnn', 'dqn'")

        # Set up loss for Q-learning
        if q_learning:
            self.rewards = ph(tf.float32, [None], name='rewards')
            self.q_values = self.probs
            self.actions = ph(tf.uint8, [None], name='action')

            Q_samp = self.rewards + self.lr * tf.reduce_max(self.q_values, axis=1)
            action_mask = tf.one_hot(indices=self.actions, depth=self.num_actions)
            self.loss = tf.reduce_sum(tf.square((Q_samp-tf.reduce_sum(self.q_values*action_mask, axis=1))))
            
        # Otherwise, set up loss for classification.
        else:
            # Define loss
            onehot_labels = tf.one_hot(self.y, self.num_actions)
            total_loss = tf.losses.softmax_cross_entropy(onehot_labels, logits=self.probs)
            self.loss = tf.reduce_mean(total_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(self.lr) # Select optimizer and set learning rate
        self.train_step = optimizer.minimize(self.loss)

    #############################
    # RUN GRAPH
    #############################

    def run_classification(self,
                           data_manager,
                           session,
                           epochs=1,
                           training_now=False,
                           validate_incrementally=False,
                           print_every=100,
                           plot_losses=False,
                           plot_accuracies=False,
                           results_dir=""):
        iter_cnt = 0 # Counter for printing
        epoch_losses = []
        epoch_accuracies = []
        total_loss, total_correct = None, None
        validate_variables, validate_losses, validate_accuracies = None, None, None

        for e in range(epochs):
            # Keep track of losses and accuracy.
            correct = 0
            losses = []

            data_manager.init_epoch()

            while data_manager.has_next_batch():
                s_batch, a_batch, _, a_eval_batch = data_manager.get_next_batch()

                # Have tensorflow compute accuracy.
                # TODO BUG: When using batches, seems to compare arrs of size (batch_size,) and (total_size,)
                correct_prediction = tf.equal(tf.argmax(self.probs, 1), a_batch)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                # batch_size = yd.shape[0] # TODO: Add back batch sizes

                # Setting up variables we want to compute (and optimizing)
                # If we have a training function, add that to things we compute
                variables = [self.loss, correct_prediction, accuracy]
                if training_now:
                    variables[-1] = self.train_step

                if validate_incrementally:
                    correct_validation = tf.equal(tf.argmax(self.probs, 1), a_eval_batch)
                    validate_variables = [self.loss, correct_validation]
                    validate_losses = []
                    validate_accuracies = []

                # Create a feed dictionary for this batch
                feed_dict = {self.X: s_batch,
                             self.y: a_batch,
                             self.is_training: training_now}

                # Get actual batch size
                # actual_batch_size = yd[idx].shape[0]
                actual_batch_size = data_manager.batch_size

                # Have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                # Aggregate performance stats
                losses.append(loss * actual_batch_size)
                correct += np.sum(corr)

                # Print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"
                          .format(iter_cnt, loss, np.sum(corr) * 1.0 / actual_batch_size))
                iter_cnt += 1

            total_correct = correct * 1.0 / data_manager.s_train.shape[0]
            total_loss = np.sum(losses) / data_manager.s_train.shape[0]
            epoch_losses.append(total_loss)
            epoch_accuracies.append(total_correct)
            print("Epoch {2}, Overall training loss = {0:.3g} and accuracy of {1:.3g}"
                  .format(total_loss, total_correct, e+1))

            if validate_incrementally:
                validate_feed_dict = {
                    self.X: data_manager.s_eval,
                    self.y: data_manager.a_eval,
                    self.is_training: False
                }
                loss, corr = session.run(validate_variables, feed_dict=validate_feed_dict)
                validate_losses.append(loss)
                validate_correct = np.sum(corr * 1.0 / data_manager.s_eval.shape[0])
                validate_accuracies.append(validate_correct)
                print("Epoch {2}, Validation loss = {0:.3g} and accuracy of {1:.3g}"
                  .format(loss, validate_correct, e+1))

        # # Plot
        # dt = str(datetime.datetime.now())
        # if plot_losses:
        #     plot("loss", epoch_losses, validate_incrementally, validate_losses, results_dir, dt)
        # if plot_accuracies:
        #     plot("accuracy", epoch_accuracies, validate_incrementally, validate_accuracies, results_dir, dt)

        return total_loss, total_correct

    def run_q_learning(self,
                       data_manager,
                       session,
                       training_now=False
                       ):
        data_manager.init_epoch()

        while data_manager.has_next_batch():
            # Perform training step.
            s_batch, a_batch, r_batch, _ = data_manager.get_next_batch()
            batch_reward = sum(r_batch)

            variables = [self.loss, self.train_step]
            feed_dict = {
                self.X: s_batch,
                self.rewards: r_batch,
                self.actions: a_batch,
                self.is_training: training_now}
            loss = session.run(variables, feed_dict=feed_dict)

            print("loss: ", loss)
            print("batch reward: ", batch_reward)

def plot(plot_name, train, validate_incrementally, validate, results_dir, dt):
    train_line = plt.plot(train, label="Training " + plot_name)
    if validate_incrementally:
        validate_line = plt.plot(validate, label="Validation " + plot_name)
    plt.legend()
    plt.grid(True)
    plt.title(plot_name)
    plt.xlabel('epoch number')
    plt.ylabel('epoch ' + plot_name)
    plt.savefig(results_dir + plot_name + "/" + dt + ".png")
    plt.close()

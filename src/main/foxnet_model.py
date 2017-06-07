# Tensorflow model declarations

import datetime
from foxnet import FoxNet
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tqdm import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
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
        if model == "dqn_3d":
            self.X = ph(tf.float32, [None, frames_per_state, height, width, n_channels])
        else:
            self.X = ph(tf.float32, [None, height, width, n_channels])
        self.y = ph(tf.int64, [None])
        self.is_training = ph(tf.bool)

        foxnet = FoxNet()

        # Build net
        if model == "fc":
            self.probs = foxnet.fully_connected(self.X, self.y, self.num_actions)
        elif model == "simple_cnn":
            self.probs = foxnet.simple_cnn(self.X, self.y, cnn_filter_size, cnn_n_filters, self.num_actions, self.is_training)
        elif model == "dqn":
            self.probs = foxnet.DQN(self.X, self.y, self.num_actions)
        elif model == "dqn_3d":
            self.probs = foxnet.DQN_3D(self.X, self.y, self.num_actions, frames_per_state)
        else:
            raise ValueError("Invalid model specified. Valid options are: 'fcc', 'simple_cnn', 'dqn', 'dqn_3d'")

        # Set up loss for Q-learning
        if q_learning:
            gamma = 0.99
            self.rewards = ph(tf.float32, [None], name='rewards')
            self.q_values = self.probs
            self.actions = ph(tf.uint8, [None], name='action')

            Q_samp = self.rewards + gamma * tf.reduce_max(self.q_values, axis=1)
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
                           plot=False,
                           results_dir="",
                           dt=""):
        iter_cnt = 0 # Counter for printing
        epoch_losses = []
        epoch_accuracies = []
        total_loss, total_correct = None, None

        validate_losses = []
        validate_accuracies = []

        for e in range(epochs):
            # Keep track of losses and accuracy.
            correct = 0
            losses = []

            data_manager.init_epoch()

            while data_manager.has_next_batch():
                s_batch, a_batch, _, _ = data_manager.get_next_batch()

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

                # Create a feed dictionary for this batch
                feed_dict = {self.X: s_batch,
                             self.y: a_batch,
                             self.is_training: training_now}

                # Get actual batch size
                # actual_batch_size = yd[idx].shape[0]
                actual_batch_size = s_batch.shape[0]

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
                val_loss, val_accuracy = self.run_validation(data_manager, session, False)
                validate_losses.append(val_loss)
                validate_accuracies.append(val_accuracy)
                print("         Validation       loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(val_loss, val_accuracy, e+1))

            # Update plot after every epoch (overwrites old version)
            if plot:
                make_classification_plot("loss", epoch_losses, validate_incrementally, validate_losses, results_dir, dt)
                make_classification_plot("accuracy", epoch_accuracies, validate_incrementally, validate_accuracies, results_dir, dt)

        # Write out summary stats
        print("##### TRAINING STATS ######################################")
        print("Final Loss: {0:.3g}\nFinal Accuracy: {1:.3g}".format(total_loss, total_correct))
        print("Epoch Losses: ")
        print(format_list(epoch_losses))
        print("Epoch Accuracies: ")
        print(format_list(epoch_accuracies))
        if (validate_incrementally):
            print("Validation Losses: ")
            print(format_list(validate_losses))
            print("Validation Accuracies: ")
            print(format_list(validate_accuracies))

        return total_loss, total_correct

    def run_validation(self,
                       data_manager,
                       session,
                       print_results=True):

        total_correct = 0
        losses = []

        data_manager.init_epoch(for_eval=True)
        while data_manager.has_next_batch(for_eval=True):
            s_batch, a_batch, _, _ = data_manager.get_next_batch(for_eval=True)
            batch_size = s_batch.shape[0]

            correct_validation = tf.equal(tf.argmax(self.probs, 1), a_batch)
            variables = [self.loss, correct_validation]

            feed_dict = {
                self.X: s_batch,
                self.y: a_batch,
                self.is_training: False
            }

            loss, correct = session.run(variables, feed_dict=feed_dict)

            losses.append(loss * batch_size)
            total_correct += np.sum(correct)

        accuracy = total_correct * 1.0 / data_manager.s_eval.shape[0]
        total_loss = np.sum(losses) / data_manager.s_eval.shape[0]

        if print_results:
            print("Validation loss = \t{0:.3g}\nValidation accuracy = \t{1:.3g}".format(total_loss, accuracy))

        return total_loss, accuracy

    def run_q_learning(self,
                       data_manager,
                       session,
                       epochs,
                       model_path,
                       results_dir,
                       training_now=False,
                       dt="",
                       plot=False,
                       plot_every=20,
                       ):

        losses = []
        scores = []
        xlabels = []

        total_batch_count = 0
        for e in range(epochs):
            data_manager.init_epoch()
            batch_count = 0

            while data_manager.has_next_batch():
                # Perform training step.
                s_batch, a_batch, r_batch, max_score_batch = data_manager.get_next_batch()
                batch_reward = sum(r_batch)
                actual_batch_size = data_manager.batch_size

                variables = [self.loss, self.train_step]
                feed_dict = {
                    self.X: s_batch,
                    self.rewards: r_batch,
                    self.actions: a_batch,
                    self.is_training: training_now}
                loss, _ = session.run(variables, feed_dict=feed_dict)

                print("loss: ", loss)
                print("batch reward: ", batch_reward)

                if batch_count % 100 == 0:
                    self.saver.save(session, model_path)
                    # Anneal epsilon
                    data_manager.epsilon *= 0.9

                # Plot loss every "plot_every" batches (overwrites prev plot)
                if plot and (total_batch_count % plot_every == 0):
                    losses.append(loss)
                    scores.append(max_score_batch)
                    xlabels.append(total_batch_count)
                    make_q_plot("loss", xlabels, losses, results_dir, dt)
                    make_q_plot("score", xlabels, scores, results_dir, dt)

                batch_count += 1
                total_batch_count += 1

def format_list(list):
    return "["+", ".join(["%.2f" % x for x in list])+"]"

def make_classification_plot(plot_name, train, validate_incrementally, validate, results_dir, dt):
    train_line = plt.plot(train, label="Training " + plot_name)
    if validate_incrementally:
        validate_line = plt.plot(validate, label="Validation " + plot_name)
    plt.legend()
    plt.grid(True)
    plt.title(plot_name)
    plt.xlabel('epoch number')
    plt.ylabel('epoch ' + plot_name)
    plt.savefig(results_dir + "classification_" + plot_name + "/" + plot_name + "_" + dt + ".png")
    plt.close()

# Overwrites previous plot each time
def make_q_plot(plot_name, x, y, results_dir, dt):
    line = plt.plot(x, y, label="Q " + plot_name)
    plt.legend()
    plt.grid(True)
    plt.title(plot_name)
    plt.xlabel('batch number')
    plt.ylabel(plot_name)
    plt.savefig(results_dir + "q_" + plot_name + "/" + plot_name + "_" + dt + ".png")
    plt.close()

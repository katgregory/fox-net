# Tensorflow model declarations

import datetime
from foxnet_model import *
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tqdm import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data_manager import DataManager

ph = tf.placeholder


class FoxNet(object):

    ############################
    # SET UP GRAPH
    #############################

    def __init__(self,
                session,
                model,
                q_learning,
                lr,
                reg_lambda,
                use_target_net,
                target_q_update_freq,
                height,
                width,
                n_channels,
                frames_per_state,
                available_actions,
                available_actions_names,
                cnn_filter_size,
                cnn_n_filters,
                load_model,
                load_model_dir,
                load_model_path,
                save_model,
                save_model_path,
                verbose = False):

        self.lr = lr
        self.reg_lambda = reg_lambda
        self.use_target_net = use_target_net
        self.target_q_update_freq = target_q_update_freq
        self.verbose = verbose
        self.available_actions = available_actions
        self.available_actions_names = available_actions_names
        self.num_actions = len(self.available_actions)
        self.q_learning = q_learning
        self.gamma = 0.99
        self.save_model = save_model
        self.save_model_path = save_model_path

        # Placeholders
        # The first dim is None, and gets sets automatically based on batch size fed in
        # count (in train/test set) x 480 (height) x 680 (width) x 3 (channels) x 3 (num frames)
        if model == "dqn_3d":
            self.states = ph(tf.float32, [None, frames_per_state, height, width, n_channels], name="states")
            self.states_p = ph(tf.float32, [None, frames_per_state, height, width, n_channels], name="states_p")
        else:
            self.states = ph(tf.float32, [None, height, width, n_channels], name="states")
            self.states_p = ph(tf.float32, [None, height, width, n_channels], name="states_p")
        self.actions = ph(tf.int64, [None], name="actions")
        self.is_training = ph(tf.bool, name="is_training")

        # Build net
        if model == "fc":
            self.model = FullyConnected(self.num_actions)
        elif model == "simple_cnn":
            self.model = SimpleCNN(cnn_filter_size, cnn_n_filters, self.num_actions, self.is_training)
        elif model == "dqn":
            self.model = DQN(self.num_actions)
        elif model == "dqn_3d":
            self.model = DQN3D(self.num_actions, frames_per_state)
        else:
            raise ValueError("Invalid model specified. Valid options are: 'fcc', 'simple_cnn', 'dqn', 'dqn_3d'")

        # Set up loss for Q-learning
        if q_learning:
            self.rewards = ph(tf.float32, [None], name='rewards')
            self.q_values = self.model.get_q_values_op(self.states)
            self.actions = ph(tf.uint8, [None], name='action')

            if self.use_target_net:
                self.q_values_p = self.model.get_q_values_op(self.states_p, scope='target_q')
                self.add_q_learning_update_target_op('q', 'target_q')
            else:
                self.q_values_p = self.model.get_q_values_op(self.states_p)

            self.add_q_learning_loss_op(self.q_values, self.q_values_p, self.num_actions)

        # Otherwise, set up loss for classification.
        else:
            # Define loss
            onehot_labels = tf.one_hot(self.actions, self.num_actions)
            action_probs = self.model.get_q_values_op(self.states)
            total_loss = tf.losses.softmax_cross_entropy(onehot_labels, logits=action_probs)
            self.loss = tf.reduce_mean(total_loss)

        # Regularization for all but biases
        if reg_lambda >= 0:
            variables = tf.trainable_variables()
            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name]) * reg_lambda
            self.loss += reg_loss

        if load_model:
            print('Loading model from dir: %s' % load_model_dir)
            loader = tf.train.import_meta_graph(load_model_path + '.meta')
            loader.restore(session, tf.train.latest_checkpoint(load_model_dir))
        else:
            # Define optimizer
            optimizer = tf.train.AdamOptimizer(self.lr) # Select optimizer and set learning rate
            vars_to_minimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
            self.train_step = optimizer.minimize(self.loss, var_list=vars_to_minimize)

    def add_q_learning_update_target_op(self, q_scope, target_q_scope):
        source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        assign_list = [tf.assign(target_vars[i], source_vars[i]) for i in range(len(target_vars))]
        self.update_target_op = tf.group(*assign_list)

    def add_q_learning_loss_op(self, q, target_q, num_actions):
        target = self.rewards + self.gamma * tf.reduce_max(self.q_values_p, axis=1)
        action_mask = tf.one_hot(indices=self.actions, depth=self.num_actions)
        prediction = tf.reduce_sum(self.q_values*action_mask, axis=1)
        self.loss = tf.reduce_sum(tf.square((target - prediction)))

    def update_target_params(self, session):
        session.run(self.update_target_op)

    #############################
    # RUN GRAPH
    #############################

    def run_classification(self,
                           data_manager,
                           session,
                           epochs,
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
                s_batch, a_batch, _, _, _ = data_manager.get_next_batch()

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
                feed_dict = {self.states: s_batch,
                             self.actions: a_batch,
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

            self.save(session)

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
        init = tf.global_variables_initializer()
        session.run(init)

        total_correct = 0
        losses = []

        data_manager.init_epoch(for_eval=True)
        while data_manager.has_next_batch(for_eval=True):
            s_batch, a_batch, r_batch, s_p_batch, _ = data_manager.get_next_batch(for_eval=True)
            batch_size = s_batch.shape[0]

            correct_validation = tf.equal(tf.argmax(self.model.get_q_values_op(self.states), 1), a_batch)
            variables = [self.loss, correct_validation]

            if self.q_learning:
                feed_dict = {
                    self.states: s_batch,
                    self.actions: a_batch,
                    self.rewards: r_batch,
                    self.states_p: s_p_batch,
                    self.is_training: False
                }
            else:
                feed_dict = {
                    self.states: s_batch,
                    self.actions: a_batch,
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
                       results_dir,
                       training_now=False,
                       dt="",
                       plot=False,
                       plot_every=1,
                       ):
        losses = []
        scores = []
        xlabels = []

        # Update the target params on initialization.
        self.update_target_params(session)

        total_batch_count = 0
        for e in range(epochs):
            data_manager.init_epoch()
            batch_count = 0

            while data_manager.has_next_batch():
                # Perform training step.
                s_batch, a_batch, r_batch, s_p_batch, max_score_batch = data_manager.get_next_batch()
                batch_reward = sum(r_batch)

                variables = [self.loss, self.train_step]
                feed_dict = {
                    self.states: s_batch,
                    self.actions: a_batch,
                    self.rewards: r_batch,
                    self.states_p: s_p_batch,
                    self.is_training: training_now}
                loss, _ = session.run(variables, feed_dict=feed_dict)

                print('Loss: %f' % loss)
                print('Batch reward: %f' % batch_reward)

                if self.use_target_net and (total_batch_count % self.target_q_update_freq == 0):
                    self.update_target_params(session)

                if total_batch_count % 100 == 0:
                    # Anneal epsilon
                    data_manager.epsilon *= 0.9
                    self.save(session)

                # Plot loss every "plot_every" batches (overwrites prev plot)
                if plot and total_batch_count % plot_every == 0:
                    print('Plotting: total_batch_count=%d' % total_batch_count)
                    losses.append(loss)
                    scores.append(max_score_batch)
                    xlabels.append(total_batch_count)
                    make_q_plot("loss", xlabels, losses, results_dir, dt)
                    make_q_plot("score", xlabels, scores, results_dir, dt)

                batch_count += 1
                total_batch_count += 1

    def save(self, session):
        if self.save_model:
            print("-- saving model --")
            self.saver.save(session, self.save_model_path)

def format_list(list):
    return "["+", ".join(["%.2f" % x for x in list])+"]"

def make_classification_plot(plot_name, train, validate_incrementally, validate, results_dir, dt):
    train_line = plt.plot(train, label="Training " + plot_name)
    if validate_incrementally:
        validate_line = plt.plot(validate, label="Validation " + plot_name)
    plt.legend()
    plt.grid(True)
    plt.title("Classification " + plot_name)
    plt.xlabel('Epoch number')
    plt.ylabel('Epoch ' + plot_name)
    plt.savefig(results_dir + "classification_" + plot_name + "/" + plot_name + "_" + dt + ".png")
    plt.close()

# Overwrites previous plot each time
def make_q_plot(plot_name, x, y, results_dir, dt):
    line = plt.plot(x, y, label="Q-learning " + plot_name)
    plt.legend()
    plt.grid(True)
    plt.title(plot_name)
    plt.xlabel('Batch number')
    plt.ylabel(plot_name)
    plt.savefig(results_dir + "q_" + plot_name + "/" + plot_name + "_" + dt + ".png")
    plt.close()

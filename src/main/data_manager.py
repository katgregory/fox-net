from emu_interact import FrameReader
from collections import deque
from health.health import HealthExtractor
from reward.knn_extract_reward_online import RewardExtractor
from data import load_datasets

import numpy as np
import math


class DataManager:
    def __init__(self):
        self.is_online = False

    def init_online(self, foxnet, session, batch_size, ip, image_height, image_width, epsilon):
        self.is_online = True
        self.foxnet = foxnet
        self.session = session
        self.batch_size = batch_size
        self.epsilon = epsilon

        # Initialize emulator transfers
        self.frame_reader = FrameReader(ip, image_height, image_width)
        self.health_extractor = HealthExtractor()
        self.reward_extractor = RewardExtractor()

        # Keep full image for reward extraction.
        state, full_image = self.frame_reader.read_frame()
        self.states = [state]

    def init_offline(self, use_test_set, data_params, batch_size):
        self.is_online = False

        # Load the two pertinent datasets into train_dataset and eval_dataset
        if use_test_set:
            train_dataset, eval_dataset = load_datasets('test', data_params)
        else:
            train_dataset, eval_dataset = load_datasets('dev', data_params)

        self.X_train, self.y_train = train_dataset
        self.X_eval, self.y_eval = eval_dataset

        # self.Xd = Xd
        # self.yd = yd
        self.batch_size = batch_size

    def init_epoch(self):
        self.batch_iteration = -1

        if self.is_online:
            pass
        else:
            self.train_indices = np.arange(self.X_train.shape[0])
            np.random.shuffle(self.train_indices)

    def has_next_batch(self):
        if self.is_online:
            return True
        else:
            num_batch_iterations = int(math.ceil(self.X_train.shape[0] / self.batch_size))
            return self.batch_iteration < num_batch_iterations

    def get_next_batch(self):
        self.batch_iteration += 1

        if self.is_online:
            state_list = []
            action_list = []
            reward_list = []

            state = self.states[-1]  # TODO: is this right?

            # Collect a batch.
            for i in range(self.batch_size):
                # TODO: replay memory stuff

                # Save states for batch forward pass.
                state_list.extend(state)

                feed_dict = {self.foxnet.X: state, self.foxnet.is_training: False}
                q_values_it = self.session.run(self.foxnet.probs, feed_dict=feed_dict)

                # e-greedy exploration.
                if np.random.uniform() >= self.epsilon:
                    action = np.argmax(q_values_it)
                else:
                    action = np.random.choice(np.arange(self.foxnet.num_actions))
                action_list.append(action)

                # Send action to emulator
                self.frame_reader.send_action(self.foxnet.available_actions[action])

                # Get next state
                new_state, full_image = self.frame_reader.read_frame()

                # Get health reward
                health_reward = self.health_extractor(full_image, offline=False)

                # Get score reward
                score_reward = self.reward_extractor.get_reward(full_image)

                reward = health_reward + score_reward
                reward_list.append(reward)

                # TODO: implement done mask?

                # TODO: store transition

                state = new_state

            return state_list, action_list, reward_list
        else:
            # Generate indices for the batch.
            start_idx = (self.batch_iteration * self.batch_size) % self.X_train.shape[0]
            idx = self.train_indices[start_idx: start_idx + self.batch_size]

            X_batch = self.X_train[idx, :]
            y_batch = self.y_train[idx]
            y_eval_batch = self.y_eval[idx]

            return X_batch, y_batch, y_eval_batch
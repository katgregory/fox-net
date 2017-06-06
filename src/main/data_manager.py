from emu_interact import FrameReader
from health.health import HealthExtractor
from reward.knn_extract_reward_online import RewardExtractor
from data import load_datasets
from replay_buffer import ReplayBuffer

import numpy as np
import math


class DataManager:
    def __init__(self):
        self.is_online = False

    def init_online(self, foxnet, session, batch_size, replay_buffer_size, frames_per_state, ip, image_height,
                    image_width, epsilon, user_overwrite=False):
        self.is_online = True
        self.foxnet = foxnet
        self.session = session
        self.batch_size = batch_size
        self.epsilon = epsilon

        # Allow player to overwrite for faster learning
        self.user_overwrite = user_overwrite

        # Initialize ReplayBuffer.
        self.replay_buffer = ReplayBuffer(replay_buffer_size, frames_per_state)

        # Initialize emulator transfers
        self.frame_reader = FrameReader(ip, image_height, image_width)
        self.health_extractor = HealthExtractor()
        self.reward_extractor = RewardExtractor()

        # Keep full image for reward extraction.
        frame, full_image = self.frame_reader.read_frame()
        self.frames = [frame]

        # Remember the health from the previous frame.
        self.prev_health = None

    def init_offline(self, use_test_set, data_params, batch_size):
        self.is_online = False
        self.user_overwrite = False

        # Load the two pertinent datasets into train_dataset and eval_dataset
        if use_test_set:
            train_dataset, eval_dataset = load_datasets('test', data_params)
        else:
            train_dataset, eval_dataset = load_datasets('dev', data_params)

        self.s_train, self.a_train, scores_train, h_train = train_dataset
        self.s_eval, self.a_eval, scores_test, h_test = eval_dataset

        # Compute the reward given scores and health. Currently, this just adds the two, weighting each one equally.
        self.r_train = np.add(scores_train, h_train)
        self.r_test = np.add(scores_test, h_test)

        self.batch_size = batch_size

    def init_epoch(self):
        self.batch_iteration = -1

        if self.is_online:
            pass
        else:
            self.train_indices = np.arange(self.s_train.shape[0])
            np.random.shuffle(self.train_indices)

    def has_next_batch(self):
        if self.is_online:
            return True
        else:
            num_batch_iterations = int(math.ceil(self.s_train.shape[0] / self.batch_size))
            return self.batch_iteration < num_batch_iterations

    def get_next_batch(self):
        s_batch = []
        a_batch = []
        r_batch = []
        a_eval_batch = []
        max_score_batch = 0

        self.batch_iteration += 1

        if self.is_online:

            frame = self.frames[-1]

            # Play the game for base_size frames.
            # TODO Introduce a new parameter specifying how many frames to play each time we update parameters.
            i = 0
            while i < self.batch_size or not self.replay_buffer.can_sample(self.batch_size):
                i += 1
                # Store the most recent frame and get the past frames_per_state frames that define the current state.
                replay_buffer_index = self.replay_buffer.store_frame(np.squeeze(frame))
                state = self.replay_buffer.encode_recent_observation()
                state = np.expand_dims(state, 0)

                # Get the best action to take in the current state.
                feed_dict = {self.foxnet.X: state, self.foxnet.is_training: False}
                q_values_it = self.session.run(self.foxnet.probs, feed_dict=feed_dict)

                action = 7

                if self.user_overwrite:
                    action = self.frame_reader.get_keys()

                # If in user-overwrite and player does not input, do e-greedy
                if action == 7:
                    # e-greedy exploration.
                    if np.random.uniform() >= self.epsilon:
                        action = np.argmax(q_values_it)
                    else:
                        action = np.random.choice(np.arange(self.foxnet.num_actions))

                # Send action to emulator.
                self.frame_reader.send_action(self.foxnet.available_actions[action])

                # Get the next frame.
                new_frame, full_image = self.frame_reader.read_frame()

                # Get the reward (score + health).
                score_reward = self.reward_extractor.get_reward(full_image)
                health_reward = self.health_extractor(full_image, offline=False)

                if self.prev_health and self.prev_health > 0 and health_reward == 0:
                    # Agent just died.
                    print('INFO: Agent just died. Setting health reward to -100')
                    health_reward = -100
                self.prev_health = health_reward

                reward = score_reward + health_reward
                max_score_batch = max(score_reward, max_score_batch)

                # Store the <s,a,r,s'> transition.
                # TODO Pass in True if terminal?
                self.replay_buffer.store_effect(replay_buffer_index, action, reward, False)

                frame = new_frame

            s_batch, a_batch, r_batch, _, _ = self.replay_buffer.sample(self.batch_size)
        else:
            # Generate indices for the batch.
            start_idx = (self.batch_iteration * self.batch_size) % self.s_train.shape[0]
            idx = self.train_indices[start_idx: start_idx + self.batch_size]

            s_batch = self.s_train[idx, :]
            a_batch = self.a_train[idx]
            r_batch = self.r_train[idx]
            a_eval_batch = self.a_eval[idx]

        print('Max score for current batch: %d' % max_score_batch)
        return s_batch, a_batch, r_batch, a_eval_batch, max_score_batch
from emu_interact import FrameReader
from health.health import HealthExtractor
from reward.knn_extract_reward_online import RewardExtractor
from data import load_datasets
from replay_buffer import ReplayBuffer
import numpy as np
import math


class DataManager:
    def __init__(self, verbose=False):
        self.is_online = False
        self.verbose = verbose

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

    def init_epoch(self, for_eval=False):
        self.batch_iteration = -1

        if self.is_online:
            pass
        else:
            if for_eval: # "epoch" is entire validation set
                self.epoch_indices = np.arange(self.s_eval.shape[0])
            else:
                self.epoch_indices = np.arange(self.s_train.shape[0])
            np.random.shuffle(self.epoch_indices)

    def has_next_batch(self, for_eval=False):
        if self.is_online:
            return True
        else:
            if for_eval:
                num_batch_iterations = int(math.ceil(self.s_eval.shape[0] / self.batch_size))
            else:
                num_batch_iterations = int(math.ceil(self.s_train.shape[0] / self.batch_size))
            return self.batch_iteration < num_batch_iterations

    def get_next_batch(self, for_eval=False):
        s_batch = []
        a_batch = []
        r_batch = []
        max_score_batch = 0

        self.batch_iteration += 1
        frame_skip = 5

        if self.is_online:

            frame = self.frames[-1]

            # Play the game for base_size frames.
            # TODO Introduce a new parameter specifying how many frames to play each time we update parameters.
            i = 0
            last_action_str = 'n'
            last_frame_had_invalid_score = False
            while i < self.batch_size or not self.replay_buffer.can_sample(self.batch_size):
                for j in np.arange(frame_skip):
                    self.frame_reader.send_action(last_action_str)
                    new_frame, full_image = self.frame_reader.read_frame()
                i += 1  
                # Store the most recent frame and get the past frames_per_state frames that define the current state.
                replay_buffer_index = self.replay_buffer.store_frame(np.squeeze(frame))
                state = self.replay_buffer.encode_recent_observation()
                state = np.expand_dims(state, 0)

                # Get the best action to take in the current state.
                if last_frame_had_invalid_score:
                    # We are not actually playing a level.
                    action_str = np.random.choice(['l', 'j'])
                    print('PRESSING SELECT (we\'re not playing a level, right?). Taking action: %s' % action_str)
                else:
                    feed_dict = {self.foxnet.X: state, self.foxnet.is_training: False}
                    q_values_it = self.session.run(self.foxnet.probs, feed_dict=feed_dict)

                    action_str = 'n'

                    if self.user_overwrite:
                        action_str = self.frame_reader.get_keys()

                    # If in user-overwrite and player does not input, do e-greedy
                    if action_str == 'n':
                        # e-greedy exploration.
                        if np.random.uniform() >= self.epsilon:
                            action_str = self.foxnet.available_actions[np.argmax(q_values_it)]
                        else:
                            action_str = np.random.choice(self.foxnet.available_actions)

                # Send action to emulator.
                self.frame_reader.send_action(action_str)

                # Remember this action for the next iteration.
                last_action_str = action_str

                # Determine the action we will send to the replay buffer.
                if last_frame_had_invalid_score:
                    # If the last frame was a non-level frame, pretend we just did a noop.
                    replay_buffer_str = self.foxnet.available_actions.index('n')
                else:
                    replay_buffer_str = self.foxnet.available_actions.index(action_str)

                # Get the next frame.
                new_frame, full_image = self.frame_reader.read_frame()

                # Get the reward (score + health).
                score_reward, score_is_not_digits = self.reward_extractor.get_reward(full_image)
                last_frame_had_invalid_score = score_is_not_digits
                health_reward = self.health_extractor(full_image, offline=False)

                if self.verbose:
                    print('Online reward extracted: score=%d\thealth=%d' % (score_reward, health_reward))

                if self.prev_health and self.prev_health > 0 and health_reward == 0:
                    # Agent just died.
                    if self.verbose:
                        print('INFO: Agent just died. Setting health reward to -100')
                    health_reward = -100
                self.prev_health = health_reward

                reward = score_reward + health_reward
                max_score_batch = max(score_reward, max_score_batch)

                # Store the <s,a,r,s'> transition.
                self.replay_buffer.store_effect(replay_buffer_index, replay_buffer_str, reward, False)

                frame = new_frame

            s_batch, a_batch, r_batch, _, _ = self.replay_buffer.sample(self.batch_size)
        else:
            # Choose which data to batch
            if (for_eval):
                s_to_batch = self.s_eval
                a_to_batch = self.a_eval
                r_to_batch = None
            else:
                s_to_batch = self.s_train
                a_to_batch = self.a_train
                r_to_batch = self.r_train

            # Generate indices for the batch.
            start_idx = (self.batch_iteration * self.batch_size) % s_to_batch.shape[0]
            idx = self.epoch_indices[start_idx: start_idx + self.batch_size]

            s_batch = s_to_batch[idx, :]
            a_batch = a_to_batch[idx]
            if (not for_eval):
                r_batch = r_to_batch[idx]

        # print('Max score for current batch: %d' % max_score_batch)
        return s_batch, a_batch, r_batch, max_score_batch

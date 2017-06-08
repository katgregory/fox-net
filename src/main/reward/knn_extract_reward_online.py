import cv2
import glob
import os
import math
import numpy as np
from reward import utils
from scipy.stats.stats import pearsonr
import os

Y_MIN = 65
Y_SIZE = 31
X_MIN = 47
X_SIZE = 26

class RewardExtractor():
    def __init__(self):
        template_dir = './data/reward/templates/*'
        self.templates = self.load_template_images(template_dir)

        self.template_values = []
        for template_filename, _ in self.templates:
            self.template_values.append(utils.digit_from_template_filename(template_filename))

        self.prev_reward = 0

    def load_template_images(self, dir, filter_image_flag=True):
        '''
        Returns a tuple for each template file in dir. Each tuple has the format:
        (filename, image as a numpy array)
        '''
        image_filenames = glob.glob(dir)

        images = []
        for image_filename in image_filenames:
            # Show each template image.
            # cv2.imshow(image_filename, cv2.imread(image_filename))
            # cv2.waitKey(0)

            image = cv2.imread(image_filename)

            if filter_image_flag:
                image = utils.filter_image(image)

            images.append((image_filename, image.flatten()))

        return images

    def modify_image(self, image, filter_image_flag=True):
        hundreds = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN : X_MIN + X_SIZE]
        tens = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN + X_SIZE : X_MIN + 2 * X_SIZE]
        ones = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN + 2 * X_SIZE : X_MIN + 3 * X_SIZE]

        if filter_image_flag:
            hundreds = utils.filter_image(hundreds)
            tens = utils.filter_image(tens)
            ones = utils.filter_image(ones)

        hundreds = hundreds.flatten()
        tens = tens.flatten()
        ones = ones.flatten()

        image = []
        image.append((hundreds))
        image.append((tens))
        image.append((ones))
        return image

    def classify_image(self, input_image):
        '''
        Returns the template index classification for each input image. Images are classified by returning the index of the
        template with the max similarity with the input image. Similarity can be either inner-product or pearson
        correlation.
        '''
        templates = self.templates
        input_image = self.modify_image(input_image)

        # Normalize templates and stack them side-by-side in a matrix.
        template_mat = np.zeros(shape=(templates[0][1].size, len(templates)))

        for index, (filename, template_image) in enumerate(templates):
            template_mat[:, index] = template_image.astype(float) / np.sum(template_image)

        # Classify each image.
        labels = []

        hundreds_digit, _ = utils.classify_digit_pearson(input_image[0], template_mat)
        tens_digit, _ = utils.classify_digit_pearson(input_image[1], template_mat)
        ones_digit, _ = utils.classify_digit_pearson(input_image[2], template_mat)
        labels.append((hundreds_digit, tens_digit, ones_digit))

        return labels

    def get_reward(self, input_image):
        labels = self.classify_image(input_image)

        if utils.UNCERTAIN_DIGIT in labels[0]:
            reward = utils.UNCERTAIN_DIGIT
        elif utils.NOT_A_DIGIT in labels[0]:
            reward = utils.NOT_A_DIGIT
        else:
            reward = self.template_values[labels[0][0]] * 100 + \
                     self.template_values[labels[0][1]] * 10 + \
                     self.template_values[labels[0][2]]

            # Hack to correct for various digits incorrectly replaced with a 0 in the hundreds digit.
            reward %= 100

        # Hack to correct for squished digits looking like 1's.
        if reward is utils.UNCERTAIN_DIGIT or \
                (reward > 0 and self.prev_reward > 0 and abs(self.prev_reward - reward) > 6):
            reward = self.prev_reward
        self.prev_reward = reward

        reward_value = max(reward, 0)

        return reward_value, reward == utils.NOT_A_DIGIT
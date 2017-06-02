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
        template_dir = './reward/templates/*'
        self.templates = self.load_template_images(template_dir)
        print(len(self.templates))

    def load_template_images(self, dir, filter_image_flag=True):
        '''
        Returns a tuple for each template file in dir. Each tuple has the format:
        (filename, image as a numpy array)
        '''
        image_filenames = glob.glob(dir)
        print(image_filenames)

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
        hundreds = input_image[0]
        tens = input_image[1]
        ones = input_image[2]

        labels.append((self.classify_digit_pearson(hundreds, template_mat),
                           self.classify_digit_pearson(tens, template_mat),
                           self.classify_digit_pearson(ones, template_mat)))

        return labels

    def get_reward(self, input_image):
        templates = self.templates
        labels = self.classify_image(input_image)

        template_values = []
        for template_filename, _ in templates:
            template_name = template_filename[template_filename.rfind('/') + 1:template_filename.rfind('.')]
            template_values.append(int(template_name))

        if None in labels[0]:
            reward = 0
        else:
            reward = template_values[labels[0][0]] * 100 + \
                     template_values[labels[0][1]] * 10 + \
                     template_values[labels[0][2]]

            # Hack to correct for 9's being replaced with 0's in the hundreds digit.
            if reward > 900:
                reward -= 900

        return reward

    def classify_digit_inner_product(self, input_digit, template_mat):
        return np.argmax(np.dot(input_digit, template_mat))


    def classify_digit_pearson(self, input_digit, template_mat):
        pearson_correlations = [pearsonr(input_digit, template_mat[:, index].flatten())[0] for index in
                                range(template_mat.shape[1])]
        max_pearson_correlation = np.max(pearson_correlations)
        if max_pearson_correlation < 0.5 or math.isnan(max_pearson_correlation):
            return None
        return np.argmax(pearson_correlations)
import cv2
import glob
import re
import math
import numpy as np
from scipy.stats.stats import pearsonr

# Pixel coordinates of score.
Y_MIN = 65
Y_SIZE = 31
X_MIN = 47
X_SIZE = 26

# Average digit pixel color values.
R_MEAN = 88
G_MEAN = 152
B_MEAN = 75
RGB_TOL = 15


# Classification labels.
UNCERTAIN_DIGIT = -1
NOT_A_DIGIT = -2


def classify_digit_inner_product(input_digit, template_mat):
    return np.argmax(np.dot(input_digit, template_mat)), np.dot(input_digit, template_mat)


def classify_digit_pearson(input_digit, template_mat):
    pearson_correlations = [pearsonr(input_digit, template_mat[:, index].flatten())[0] for index in
                            range(template_mat.shape[1])]

    max_pearson_correlation = np.max(pearson_correlations)

    digit = None
    if max_pearson_correlation < 0.5 or math.isnan(max_pearson_correlation):
        digit = NOT_A_DIGIT
    elif max_pearson_correlation < 0.9:
        digit = UNCERTAIN_DIGIT
    else:
        digit = np.argmax(pearson_correlations)

    return digit, pearson_correlations


def print_probs(probs, template_values):
    if probs is None or np.isnan(probs).any():
        print('Invalid pearson correlations.')
    else:
        print('\t%s' % [int(x * 100) / 100.0 for (y, x) in sorted(zip(template_values, probs))])


data_pattern = re.compile('.*i=(\d+)_a=(\w).*.png')
def iteration_from_filename(filename):
    match = re.search(data_pattern, filename)
    iteration = match.group(1)
    return int(iteration)


template_pattern = re.compile('(\d+)(_(\d+))?.png')
def digit_from_template_filename(filename):
    match = re.search(template_pattern, filename)
    digit = match.group(1)
    return int(digit)


def filter_image(image):
    image = image[:, :, 0]

    # Filter out pixels falling outside of the target range.
    # image[abs(image - B_MEAN) > RGB_TOL] = 0
    # cv2.imshow('filtered', image)
    # cv2.waitKey(0)

    return image

def load_images(dir, filter_image_flag=True, group_by_image=True):
    '''
    :param group_by_image:
       * If true, returns a tuple for each input file in dir. Each tuple has the format:
         (filename, 100s digit as numpy array, 10s digit as a numpy array, 1s digit as a numpy array)
       * If false, returns a tuple for each digit in each input file in dir. Each tuple has the format:
         (filename, digit as numpy array)
    '''

    image_filenames = [filename for _, filename in
                       sorted([(iteration_from_filename(filename), filename) for filename in glob.glob(dir)])]

    images = []
    for index, image_filename in enumerate(image_filenames):
        image = cv2.imread(image_filename)
        hundreds = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN : X_MIN + X_SIZE]
        tens = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN + X_SIZE : X_MIN + 2 * X_SIZE]
        ones = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN + 2 * X_SIZE : X_MIN + 3 * X_SIZE]

        if filter_image_flag:
            hundreds = filter_image(hundreds)
            tens = filter_image(tens)
            ones = filter_image(ones)

        # cv2.imwrite(str(index) + '_100.png', hundreds)
        # cv2.imwrite(str(index) + '_10.png', tens)
        # cv2.imwrite(str(index) + '_1.png', ones)

        hundreds = hundreds.flatten()
        tens = tens.flatten()
        ones = ones.flatten()

        if group_by_image:
            images.append((image_filename, hundreds, tens, ones))
        else:
            images.append((image_filename, hundreds))
            images.append((image_filename, tens))
            images.append((image_filename, ones))

    return images
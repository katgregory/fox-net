import cv2
import glob
import os
import math
import numpy as np
import util
from scipy.stats.stats import pearsonr
from optparse import OptionParser


def get_options():
    '''
    Get command-line options and handle errors.
    :return: Command line options and arguments.
    '''
    parser = OptionParser()

    parser.add_option('-t', '--template_dir', dest='template_dir',
                      help='Directory with a template image for each digit.')

    parser.add_option('-i', '--input_dir', dest='input_dir',
                      help='Directory with input images to be classified.')

    parser.add_option('-o', '--output_dir', dest='output_dir',
                      help='(optional) If specified, saves images to output_dir instead of renaming them.')

    options, args = parser.parse_args()
    return options, args


def load_template_images(dir, filter_image_flag=True):
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
            image = util.filter_image(image)

        images.append((image_filename, image.flatten()))

    return images


def classify_images(templates, input_images):
    '''
    Returns the template index classification for each input image. Images are classified by returning the index of the
    template with the max similarity with the input image. Similarity can be either inner-product or pearson
    correlation.
    '''

    # Normalize templates and stack them side-by-side in a matrix.
    template_mat = np.zeros(shape=(templates[0][1].size, len(templates)))

    for index, (filename, template_image) in enumerate(templates):
        template_mat[:, index] = template_image.astype(float) / np.sum(template_image)

    # Classify each image.
    labels = []
    for _, hundreds, tens, ones in input_images:
        labels.append((classify_digit_pearson(hundreds, template_mat),
                       classify_digit_pearson(tens, template_mat),
                       classify_digit_pearson(ones, template_mat)))

    return labels


def classify_digit_inner_product(input_digit, template_mat):
    return np.argmax(np.dot(input_digit, template_mat))


def classify_digit_pearson(input_digit, template_mat):
    pearson_correlations = [pearsonr(input_digit, template_mat[:, index].flatten())[0] for index in
                            range(template_mat.shape[1])]
    max_pearson_correlation = np.max(pearson_correlations)
    if max_pearson_correlation < 0.5 or math.isnan(max_pearson_correlation):
        return None
    return np.argmax(pearson_correlations)


def save_input_images(templates, input_images, labels, output_dir=None):
    '''
    If output_dir is none, renames each original input image with its reward appended. Otherwise, saves a new copy of
    each image with its reward appended.
    '''
    template_values = []
    for template_filename, _ in templates:
        template_name = template_filename[template_filename.rfind('/') + 1:template_filename.rfind('.')]
        template_values.append(int(template_name))

    if output_dir and '/' not in output_dir:
        output_dir += '/'

    prev_reward = 0

    for index, (input_filename, _, _, _) in enumerate(input_images):
        input_base = input_filename[:input_filename.rfind('.')]
        input_extension = input_filename[input_filename.rfind('.'):]
        input_name = input_base[input_base.rfind('/') + 1:]

        if None in labels[index]:
            reward = 0
        else:
            reward = template_values[labels[index][0]] * 100 + \
                     template_values[labels[index][1]] * 10 + \
                     template_values[labels[index][2]]

            # Hack to correct for 9's being replaced with 0's in the hundreds digit.
            if reward > 900:
                reward -= 900

        # Hack to correct for squished digits looking like 1's.
        if prev_reward > 0 and (prev_reward > reward or reward - prev_reward > 10):
            reward = prev_reward

        if output_dir is None:
            os.rename(input_filename, input_base + '_' + str(reward) + input_extension)
        else:
            cv2.imwrite(output_dir + input_name + '_r=' + str(reward) + input_extension, cv2.imread(input_filename))

        prev_reward = reward


def main():
    # Parse the command line arguments.
    options, args = get_options()

    # Load template and input images.
    templates = load_template_images(options.template_dir)
    input_images = util.load_images(options.input_dir)

    # Classify each input image.
    labels = classify_images(templates, input_images)

    # Save each input image with its reward.
    save_input_images(templates, input_images, labels, options.output_dir)


if __name__ == '__main__':
    main()
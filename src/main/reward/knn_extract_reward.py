import cv2
import glob
import os
import numpy as np
import utils
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

    parser.add_option('--overwrite', dest='overwrite', action='store_true',
                      help='(optional) If specified, replaces the s_=\\d at the end of each filename.')

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
            image = utils.filter_image(image)

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
    labels_probs = []
    for _, hundreds, tens, ones in input_images:
        h, hp = utils.classify_digit_pearson(hundreds, template_mat)
        t, tp = utils.classify_digit_pearson(tens, template_mat)
        o, op = utils.classify_digit_pearson(ones, template_mat)
        labels.append((h, t, o))
        labels_probs.append((hp, tp, op))

    return labels, labels_probs


def save_input_images(templates, input_images, labels, labels_probs=None, output_dir=None, overwrite=False):
    '''
    If output_dir is none, renames each original input image with its reward appended. Otherwise, saves a new copy of
    each image with its reward appended.
    '''
    template_values = []
    for template_filename, _ in templates:
        template_values.append(utils.digit_from_template_filename(template_filename))
    print('Template values: %s' % str(template_values))

    if output_dir and '/' not in output_dir:
        output_dir += '/'

    prev_reward = 0

    for index, (input_filename, _, _, _) in enumerate(input_images):
        input_base = input_filename[:input_filename.rfind('.')]
        input_extension = input_filename[input_filename.rfind('.'):]
        input_name = input_base[input_base.rfind('/') + 1:]

        if utils.UNCERTAIN_DIGIT in labels[index]:
            reward = utils.UNCERTAIN_DIGIT
        elif utils.NOT_A_DIGIT in labels[index]:
            reward = utils.NOT_A_DIGIT
        else:
            reward = template_values[labels[index][0]] * 100 + \
                     template_values[labels[index][1]] * 10 + \
                     template_values[labels[index][2]]

            # Hack to correct for various digits incorrectly replaced with a 0 in the hundreds digit.
            reward %= 150

        # Hack to correct for squished digits looking like 1's.
        if reward is utils.UNCERTAIN_DIGIT or \
                (reward > 0 and prev_reward > 0 and abs(prev_reward - reward) > 6):
            reward = prev_reward
        prev_reward = reward

        reward_value = max(reward, 0)

        if output_dir is None:
            if overwrite and '_s=' in input_base:
                input_base = input_base[:input_base.rfind('_s=')]
            output_filename = input_base + '_s=' + str(reward_value) + input_extension
            os.rename(input_filename, output_filename)
        else:
            if overwrite and '_s=' in input_name:
                input_name = input_name[:input_name.rfind('_s=')]
            output_filename = output_dir + input_name + '_s=' + str(reward_value) + input_extension
            cv2.imwrite(output_filename, cv2.imread(input_filename))

        print(output_filename)
        utils.print_probs(labels_probs[index][0], template_values)
        utils.print_probs(labels_probs[index][1], template_values)
        utils.print_probs(labels_probs[index][2], template_values)

def main():
    # Parse the command line arguments.
    options, args = get_options()

    # Load template and input images.
    templates = load_template_images(options.template_dir)
    input_images = utils.load_images(options.input_dir)

    # Classify each input image.
    labels, labels_probs = classify_images(templates, input_images)

    # Save each input image with its reward.
    save_input_images(templates, input_images, labels, labels_probs, options.output_dir, options.overwrite)


if __name__ == '__main__':
    main()
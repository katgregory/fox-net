import cv2
import glob
import os
import numpy as np
from optparse import OptionParser


Y_MIN = 65
Y_SIZE = 31
X_MIN = 47
X_SIZE = 26


def get_options():
    '''
    Get command-line options and handle errors.
    :return: Command line options and arguments.
    '''
    parser = OptionParser()

    parser.add_option('-t', '--template_dir', dest='template_dir',
                      help='directory with a template image for each digit')

    parser.add_option('-i', '--input_dir', dest='input_dir',
                      help='directory with input images to be classified')

    options, args = parser.parse_args()
    return options, args


def load_template_images(dir):
    '''
    Returns a tuple for each template file in dir. Each tuple has the format:
    (filename, image as a numpy array)
    '''
    image_filenames = glob.glob(dir)
    images = [(image_filename, cv2.imread(image_filename).flatten()) for image_filename in image_filenames]
    return images


def load_input_images(dir):
    '''
    Returns a tuple for each input file in dir. Each tuple has the format:
    (filename, 100s digit as numpy array, 10s digit as a numpy array, 1s digit as a numpy array)
    '''
    image_filenames = glob.glob(dir)
    images = []
    for index, image_filename in enumerate(image_filenames):
        image = cv2.imread(image_filename)
        hundreds = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN : X_MIN + X_SIZE]
        tens = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN + X_SIZE : X_MIN + 2 * X_SIZE]
        ones = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN + 2 * X_SIZE : X_MIN + 3 * X_SIZE]

        # cv2.imwrite(str(index) + '_100.png', hundreds)
        # cv2.imwrite(str(index) + '_10.png', tens)
        # cv2.imwrite(str(index) + '_1.png', ones)

        images.append((image_filename, hundreds.flatten(), tens.flatten(), ones.flatten()))
    return images


def classify_images(templates, input_images):
    '''
    Returns the template index classification for each input image. Images are classified by returning the index of the
    template with the max dot product with the input image.
    '''

    # Normalize templates and stack them side-by-side in a matrix.
    template_mat = np.zeros(shape=(templates[0][1].size, len(templates)))

    for index, (filename, template_image) in enumerate(templates):
        template_mat[:, index] = template_image.astype(float) / np.sum(template_image)

    # Classify each image.
    labels = []
    for _, hundreds, tens, ones in input_images:
        labels.append((classify_digit(hundreds, template_mat),
                       classify_digit(tens, template_mat),
                       classify_digit(ones, template_mat)))

    return labels


def classify_digit(input_digit, template_mat):
    return np.argmax(np.dot(input_digit, template_mat))


def rename_input_images(templates, input_images, labels):
    template_values = []
    for template_filename, _ in templates:
        template_name = template_filename[template_filename.rfind('/') + 1:template_filename.rfind('.')]
        template_values.append(int(template_name))

    for index, (input_filename, _, _, _) in enumerate(input_images):
        input_base = input_filename[:input_filename.rfind('.')]
        input_extension = input_filename[input_filename.rfind('.'):]
        reward = template_values[labels[index][0]] * 100 + \
                 template_values[labels[index][1]] * 10 + \
                 template_values[labels[index][2]]
        os.rename(input_filename, input_base + '_' + str(reward) + input_extension)


def main():
    # Parse the command line arguments.
    options, args = get_options()

    # Load template and input images.
    templates = load_template_images(options.template_dir)
    input_images = load_input_images(options.input_dir)

    # Classify each input image.
    labels = classify_images(templates, input_images)

    # Rename each input image by appending its classification.
    rename_input_images(templates, input_images, labels)

if __name__ == '__main__':
    main()
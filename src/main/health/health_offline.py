import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
from health import HealthExtractor
from optparse import OptionParser
import glob
from scipy.misc import imread, imsave

def iteration_from_filename(filename):
    name = filename[filename.rfind('/i=') + 3:]
    iteration = name[:name.find('_')]
    return int(iteration)

def load_images(dir, extractor):
    image_filenames = [filename for _, filename in sorted([(iteration_from_filename(filename), filename) for filename in glob.glob(dir)])]
    return image_filenames

def get_options():
    '''
    Get command-line options and handle errors.
    :return: Command line options and arguments.
    '''
    parser = OptionParser()

    parser.add_option('-i', '--input_dir', dest='input_dir',
                      help='Directory with input images to be classified.')

    parser.add_option('-o', '--output_dir', dest='output_dir',
                      help='(optional) If specified, saves images to output_dir instead of renaming them.')

    options, args = parser.parse_args()
    return options, args

if __name__ == '__main__':
    # Parse the command line arguments
    options, args = get_options()
    output_dir = options.output_dir
    if output_dir and '/' not in output_dir:
        output_dir += '/'

    # Create health extractor
    heo = HealthExtractor('./healthbar.png', './red_frame.png')

    image_filenames = load_images(options.input_dir, heo)
    for index, input_filename in enumerate(image_filenames):
        health_ratio = heo(input_filename)

        input_base = input_filename[:input_filename.rfind('.')]
        input_extension = input_filename[input_filename.rfind('.'):]
        input_name = input_base[input_base.rfind('/') + 1:]
        print(input_filename, health_ratio)
        if output_dir is None:
            os.rename(input_filename, input_base + '_h=' + str(health_ratio) + input_extension)
        else:
            imsave(output_dir + input_name + '_h=' + str(health_ratio) + input_extension, imread(input_filename))
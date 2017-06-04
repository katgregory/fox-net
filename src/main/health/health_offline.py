import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
from health import HealthExtractor
from optparse import OptionParser
import glob
from scipy.misc import imread

# class HealthExtractorOffline(HealthExtractor):
# 	def __init__(self):
# 		self.ul = [42, 56]
# 		self.br = [53, 149]
# 		self.totalpixels = (self.br[1]-self.ul[1])*(self.br[0]-self.ul[0])
# 		self.maxhealth = imread('./health/healthbar.png').astype(float)
# 		self.thresh = 25

# 	def __call__(self, input_image, offline=True):
# 		if offline:
# 			image = imread(input_image)
# 		else:
# 			# switch BGR to RGB
# 			image = input_image[..., [2,1,0]]

# 		self.get_cur_healthbar(image)

# 		roll = np.random.uniform()
# 		return self.compare_health()

# 	def get_cur_healthbar(self, image):
# 		self.curhealth = image[self.ul[0]:self.br[0], self.ul[1]:self.br[1], :].astype(float)

# 	def compare_health(self):
# 		absdiff = np.abs(np.linalg.norm(self.maxhealth-self.curhealth, axis=2))
# 		health_sum = np.sum(absdiff <= self.thresh)
# 		health_ratio = health_sum/self.totalpixels
# 		return health_ratio

def iteration_from_filename(filename):
    name = filename[filename.rfind('/') + 1:]
    iteration = name[:name.find('_')]
    return int(iteration)

def load_images(dir):
	image_filenames = [filename for _, filename in
                       sorted([(iteration_from_filename(filename), filename) for filename in glob.glob(dir)])]
    print(image_filenames)


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
	input_images = load_images(options.input_dir)

	# Create health extractor
	heo = HealthExtractor()

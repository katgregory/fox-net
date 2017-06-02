import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt

class HealthExtractor():
	def __init__(self):
		self.ul = [42, 56]
		self.br = [53, 149]
		self.totalpixels = (self.br[1]-self.ul[1])*(self.br[0]-self.ul[0])
		self.maxhealth = imread('/home/kevin/Desktop/fox-net/src/main/health/healthbar.png').astype(float)
		self.thresh = 25

	def __call__(self, input_image, offline=True):
		if offline:
			image = imread(input_image).astype(float)
		else:
			image = input_image.astype(float)

		self.get_cur_healthbar(image)
		return self.compare_health()

	def get_cur_healthbar(self, image):
		self.curhealth = image[self.ul[0]:self.br[0], self.ul[1]:self.br[1], :].astype(float)

	def compare_health(self):
		absdiff = np.abs(np.linalg.norm(self.maxhealth-self.curhealth, axis=2))
		health_sum = np.sum(absdiff <= self.thresh)
		health_ratio = health_sum/self.totalpixels
		return health_ratio
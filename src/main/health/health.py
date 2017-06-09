import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt

class HealthExtractor():
    def __init__(self, healthbar_path='./src/main/health/healthbar.png', 
                 red_frame_path='./src/main/health/red_frame.png'):
        self.ul = [42, 56]
        self.br = [53, 149]
        self.totalpixels = (self.br[1]-self.ul[1])*(self.br[0]-self.ul[0])
        self.maxhealth = imread(healthbar_path).astype(float)
        self.thresh = 25
        self.red_frame_ave_r_value = np.mean(imread(red_frame_path)[:, :, 0])
        self.ave_r_value_threshold = 50
        self.prev_health = 0

    def __call__(self, input_image, offline=True):
        if offline:
            image = imread(input_image)
        else:
            # switch BGR to RGB
            image = input_image[..., [2,1,0]]

        if self.is_red_frame(image):
            print('Health extractor: Frame is red, meaning the agent was just hit. Returning previous health.')
            return self.prev_health

        self.get_cur_healthbar(image)

        health = self.compare_health()
        self.prev_health = health
        return health

    def get_cur_healthbar(self, image):
        self.curhealth = image[self.ul[0]:self.br[0], self.ul[1]:self.br[1], :].astype(float)

    def compare_health(self):
        absdiff = np.abs(np.linalg.norm(self.maxhealth-self.curhealth, axis=2))
        health_sum = np.sum(absdiff <= self.thresh)
        health_ratio = float(health_sum)/self.totalpixels

        health_ratio = np.round(health_ratio, 3)
        # Hack to prevent background from registering on non-health screens
        if health_ratio <= 0.05:
            health_ratio = 0.0

        health_ratio = int(10*health_ratio)
        return health_ratio

    def is_red_frame(self, image):
        curr_ave_r_value = np.mean(image[:, :, 0])
        return curr_ave_r_value > (self.red_frame_ave_r_value - self.ave_r_value_threshold)
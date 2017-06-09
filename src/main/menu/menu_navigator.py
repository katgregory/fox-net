import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imshow

class MenuNavigator:

    def __init__(self, menu_image_pathname='./src/main/menu/menu_image.png'):
        self.menu_options = self._extract_menu_options_from_image(imread(menu_image_pathname))

    def is_image_menu(self, image):
        # Switch BGR to RGB.
        image = image[..., [2, 1, 0]]
        diff = np.sum(self._extract_menu_options_from_image(image) - self.menu_options)
        ratio = float(diff) / (self.menu_options.size * 255.0)
        return ratio < 0.01

    def _extract_menu_options_from_image(self, image):
        return image[100:380, 190:440, :]
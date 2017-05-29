import cv2
import numpy as np

class FrameDisplayer():
	def __init__(self):
		pass

	def __call__(self):
		pass

	def display_frame(self, frame):
		frame = frame.squeeze()
		cv2.imshow("Input Frame", frame)
		cv2.waitKey(1)
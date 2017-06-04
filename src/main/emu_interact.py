import socket
import numpy as np
import os, sys, struct
import time
from scipy.misc import imresize
from matplotlib import pyplot as plt


class FrameReader():
	def __init__(self, ip, out_height, out_width):
		self.WIDTH = 640
		self.HEIGHT = 480
		self.DEPTH = 3
		self.BUF_SIZE = (self.WIDTH*self.HEIGHT*self.DEPTH)
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

		self.out_height = out_height
		self.out_width = out_width
		self.ip = ip
		self._launch_socket()

	def _launch_socket(self):
		# Wait for the emulator to connect
		for i in range(100):
			try:
				# Use socket.gethostname() for local server
				# Specify host IP for remote
				self.s.connect((self.ip, 11111))
				print('Socket connected successfully')
				break
			except:
				time.sleep(0.2)
				print("Retrying socket: " + str(i) + "/100")

	def read_frame(self):
		toread = self.BUF_SIZE
		buf = bytearray(self.BUF_SIZE)
		view = memoryview(buf)

		while toread:
			nbytes = self.s.recv_into(view, toread)
			view = view[nbytes:]
			toread -= nbytes

		img_flat = np.frombuffer(buf[::-1], dtype=np.uint8)
		full_img = np.reshape(img_flat, (self.HEIGHT, self.WIDTH, self.DEPTH))[:,::-1,:]
		img = imresize(full_img, (self.out_height, self.out_width, self.DEPTH))

		img = np.expand_dims(img, axis=0)

		return img, full_img

	def send_action(self, action):
		action_map = {
			'w': 0x50000000, 
			'a': 0xb00000, 
			's': 0xb0000000, 
			'd': 0x500000, 
			'j': 0x80, 
			'k': 0x40,
			'l': 0x10,
			'n': 0
		}
		self.s.send(struct.pack('I',action_map[action]))
import socket
import cv2
import numpy as np
import os, sys, struct

import time


class FrameReader():
	def __init__(self):
		self.WIDTH = 640
		self.HEIGHT = 480
		self.DEPTH = 3
		self.BUF_SIZE = (self.WIDTH*self.HEIGHT*self.DEPTH)
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

		self._launch_socket()

	def _launch_socket(self):
		# Wait for the emulator to connect
		for i in range(100):
			print(i)
			try:
				self.s.connect(("localhost", 11111))
				print('Socket connected successfully')
				break
			except:
				print(".")
				time.sleep(0.2)

	def read_frame(self):
		toread = self.BUF_SIZE
		buf = bytearray(self.BUF_SIZE)
		view = memoryview(buf)

		while toread:
			nbytes = self.s.recv_into(view, toread)
			view = view[nbytes:]
			toread -= nbytes

		img_flat = np.frombuffer(buf[::-1], dtype=np.uint8)
		img = np.reshape(img_flat, (self.HEIGHT, self.WIDTH, self.DEPTH))[:,::-1,:]
		return img

	def send_action(self, action):
		action_map = {
			'w': 0x50000000, 
			'a': 0xb00000, 
			's': 0xb0000000, 
			'd': 0x500000, 
			'j': 0x80, 
			# 'l': 0x10,
			'l': 0, 
			'n': 0
		}
		self.s.send(struct.pack('I',action_map[action]))
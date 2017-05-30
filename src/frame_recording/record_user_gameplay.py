import socket
import cv2
import numpy as np
import os, sys, struct

import keylogger.keylogger as keylogger
import time

### Globals ###
WIDTH=640
HEIGHT=480
DEPTH=3
BUF_SIZE=(WIDTH*HEIGHT*DEPTH)

# Create the socket for communicating with the emulator
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

modifier_mappings = {
	'left shift': 'a',
	'right shift': 'd',
	'left alt': 'w',
	'right alt': 's',
	'left ctrl': 'n',
	'right ctrl': 'n',
}


def launch_socket():
	# Wait for the emulator to connect
	for i in range(100):
		print(i)
		try:
			s.connect(("localhost", 11111))
			print('success')
			break
		except:
			print(".")
			time.sleep(0.2)


def read_frame():
	toread = BUF_SIZE
	buf = bytearray(BUF_SIZE)
	view = memoryview(buf)

	while toread:
		nbytes = s.recv_into(view, toread)
		view = view[nbytes:]
		toread -= nbytes

	img_flat = np.frombuffer(buf[::-1], dtype=np.uint8)
	img = np.reshape(img_flat, (HEIGHT, WIDTH, DEPTH))[:,::-1,:]
	return img


def get_keys():
	keys, modifiers = keylogger.log2()

	# If no key pressed, set to 'n'
	if keys == None:
		keys = 'n'
	# Don't get movement if shootsudo 
	elif keys == 'j':
		return keys

	# Get movement as modifiers, as these register key holds
	for mod in modifiers.keys():
		if modifiers[mod] == True:
			keys = modifier_mappings[mod]

	

	return keys


def save_frame(frame, keys, frame_num):
	filename = 'i=' + str(frame_num) + '_a=' + str(keys) + '.png'
	cv2.imwrite('./' + filename, frame)


def make_output_dir():
	date_time = time.localtime(time.time())
	year = date_time[0]
	month = date_time[1]
	day = date_time[2]
	hour = date_time[3]
	minute = date_time[4]

	path = '../../data/' + str(month) + str(day) + str(year) + '_' + str(hour) + str(minute)
	try:
		os.mkdir(path)
		os.chdir(path)
	except:
		print("Output directory already exists. Terminating script.")
		sys.exit()


def update_frame_num(frame_num):
	return frame_num + 1


def save_frame_decision(keys):
	if keys == 'n':
		roll = np.random.uniform()
		return roll > 0.95
	else:
		return True

def main():
	frame_num = 0

	launch_socket()
	make_output_dir()
	print("Made dir")

	while True:
		# Grab the frame
		frame = read_frame()
		# Grab keypresses
		keys = get_keys()
		print(keys)
		# Emulator requires a value be sent back over socket
		s.send(struct.pack('I',0))
		# Save labeled image

		# if save_frame_decision(keys):
		save_frame(frame, keys, frame_num)

		# Update frame counter
		frame_num = update_frame_num(frame_num)


if __name__ == '__main__':
	main()
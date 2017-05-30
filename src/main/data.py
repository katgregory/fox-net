# Parcel data into usable format

from collections import Counter
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from scipy import ndimage
from scipy import misc
from tqdm import *

def load_datasets(tier, params):
    print("##### LOADING DATA ########################################")
    # Matching group 1 is frame number, matching group 2 is action
    pattern = re.compile('i=(\d+)_a=(.*).png')

    # Load images
    print("Loading images:")
    images = []
    labels = []
    action_counter = Counter()

    print(params['actions'])

    for dirname in tqdm(os.listdir(params["data_dir"])):
        for filename in tqdm(os.listdir(params["data_dir"] + '/' + dirname)):
            # Stop early if have enough images
            if params["num_images"] != -1 and len(images) >= params["num_images"]:
                break

            # Extract metadata
            match = re.search(pattern, filename)
            frame_number = match.group(1)
            action = match.group(2)
            # Convert image and add to collection
            # Shape of img is (480, 640, 3)

            img = ndimage.imread(params["data_dir"] + dirname + '/' + filename)
            img = misc.imresize(img, (params['height'], params['width']))

            if img is not None and action in params["actions"]:
                images.append(img)
                labels.append(params["actions"].index(action))
                action_counter[action] += 1
    print("Loaded " + str(len(images)) + " images.")
    print("Action counts: " + str(action_counter))

    # Create states by adding a third dimension over n_frames frames
    # Final state has shape (48, 64, 3, n_frames)
    # We discard the first (n_frames - 1) frames
    if (params["multi_frame_state"]):
        print("Creating states:")
        n_frames = params["frames_per_state"]
        states = []
        for i in tqdm(xrange(len(images) - n_frames + 1)):
            state = tuple(images[x][:, :, :, None] for x in xrange(i, i + n_frames - 1))
            states.append(np.concatenate(state, axis=3))
        labels = labels[n_frames - 1:] # Also remove (n_frames - 1) labels
        print("Created " + str(len(states)) + " states.")
    else: # Final state has shape (480, 640, 3)
        states = images

    # Partition
    X_train, X_test, y_train, y_test = train_test_split(states, labels, test_size=params["eval_proportion"], random_state=42)
    print("Train count: " + str(len(X_train)) + ", Test count: " + str(len(X_test)))

    # Convert from list to numpy array
    print("Stacking...")
    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    y_train = np.stack(y_train)
    y_test = np.stack(y_test)

    # Print stats
    # Data: count (in train/test set) x 48 (height) x 68 (width) x 3 (channels) [x 3 (num frames) if multi_frame_state]
    # Labels: count (in train/test set) x 1 (index of action in ACTIONS array)
    print('Train data shape: ' + str(X_train.shape))
    print('Train labels shape: ' + str(y_train.shape))
    print('Test data shape: ' + str(X_test.shape))
    print('Test labels shape: ' + str(y_test.shape))
    print("##### DONE LOADING DATA ###################################")

    # Returns
    return (X_train, y_train), (X_test, y_test)

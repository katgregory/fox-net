# Parcel data into usable format

import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from scipy import ndimage
from tqdm import *

def load_datasets(tier, params):
    print("##### LOADING DATA ########################################")
    # Matching group 1 is frame number, matching group 2 is action
    pattern = re.compile('(\d+)_(.*).png')

    # Load images
    print("Loading images:")
    images = []
    labels = []
    for filename in tqdm(os.listdir(params["data_dir"])):
        # Stop early if have enough images
        if params["num_images"] != -1 and len(images) >= params["num_images"]:
            break

        # Extract metadata
        match = re.search(pattern, filename)
        frame_number = match.group(1)
        action = match.group(2)

        # Convert image and add to collection
        # Shape of img is (480, 640, 3)
        img = ndimage.imread(params["data_dir"] + filename)
        if img is not None:
            images.append(img)
            labels.append(action)
    print("Loaded " + str(len(images)) + " images.")

    # Create states by adding a third dimension over n_frames frames
    # Final state has shape (480, 640, 3, n_frames)
    # We discard the first (n_frames - 1) frames
    print("Creating states:")
    n_frames = params["frames_per_state"]
    states = []
    for i in tqdm(xrange(len(images) - n_frames + 1)):
        state = tuple(images[x][:, :, :, None] for x in xrange(i, i + n_frames))
        states.append(np.concatenate(state, axis=3))
    labels = labels[n_frames - 1:] # Also remove (n_frames - 1) labels
    print("Created " + str(len(states)) + " states.")

    # Partition
    X_train, X_test, y_train, y_test = train_test_split(states, labels, test_size=params["eval_proportion"], random_state=42)
    print("Train count: " + str(len(X_train)) + ", Test count: " + str(len(X_test)))

    # Convert from list to numpy array
    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    y_train = np.stack(y_train)
    y_test = np.stack(y_test)

    # Print stats
    print('Train data shape: ' + str(X_train.shape))
    print('Train labels shape: ' + str(y_train.shape))
    print('Test data shape: ' + str(X_test.shape))
    print('Test labels shape: ' + str(y_test.shape))
    print("##### DONE LOADING DATA ###################################")

    # Returns
    return (X_train, y_train), (X_test, y_test)

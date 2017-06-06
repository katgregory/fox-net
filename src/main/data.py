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
    pattern = re.compile('i=(\d+)_a=(\w)(_s=(\d+)_h=(\d+))?.png')

    # Load images
    print("Loading images:")
    images = []
    actions = []
    scores = []
    healths = []
    action_counter = Counter()

    print("Actions: " + str(params['actions']))

    for dirname in tqdm(os.listdir(params["data_dir"])):
        for filename in tqdm(os.listdir(params["data_dir"] + '/' + dirname)):
            # Stop early if have enough images
            if params["num_images"] != -1 and len(images) >= params["num_images"]:
                break

            # Extract metadata
            match = re.search(pattern, filename)
            frame_number = match.group(1)
            action = match.group(2)
            score = 0
            health = 0
            if match.group(3) is not None:
                score = int(match.group(4))
                health = int(match.group(5))
            
            # Convert image and add to collection
            # Shape of img is (480, 640, 3)

            img = ndimage.imread(params["data_dir"] + '/' + dirname + '/' + filename)
            img = misc.imresize(img, (params['height'], params['width']))

            if img is not None and action in params["actions"]:
                images.append(img)
                actions.append(params["actions"].index(action))
                action_counter[action] += 1
                scores.append(score)
                healths.append(health)
    print("Loaded " + str(len(images)) + " images.")
    print("Action counts: " + str(action_counter))

    # Create states by adding a third dimension over n_frames frames
    # Final state has shape (n_frames, 48, 64, 3)
    # We discard the first (n_frames - 1) frames
    if (params["multi_frame_state"]):
        print("Creating states:")
        n_frames = params["frames_per_state"]
        states = []
        for i in tqdm(np.arange(len(images) - n_frames + 1)):
            state = tuple(images[x][None, :, :, :] for x in np.arange(i, i + n_frames))
            states.append(np.concatenate(state, axis=0))
        actions = actions[n_frames - 1:] # Also remove (n_frames - 1) actions
        scores = scores[n_frames - 1:] # Also remove (n_frames - 1) actions
        healths = healths[n_frames - 1:] # Also remove (n_frames - 1) actions
        print("Created " + str(len(states)) + " states.")
    else: # Final state has shape (480, 640, 3)
        states = images

    # Partition
    s_train, s_test, a_train, a_test, scores_train, scores_test, h_train, h_test = \
        train_test_split(states, actions, scores, healths, test_size=params["eval_proportion"], random_state=42)
    print("Train count: " + str(len(s_train)) + ", Test count: " + str(len(s_test)))

    # Convert from list to numpy array
    print("Stacking...")
    s_train = np.stack(s_train)
    s_test = np.stack(s_test)
    a_train = np.stack(a_train)
    a_test = np.stack(a_test)
    scores_train = np.stack(scores_train)
    scores_test = np.stack(scores_test)
    h_train = np.stack(h_train)
    h_test = np.stack(h_test)

    # Print stats
    # Data: count (in train/test set) [x 3 (num frames) if multi_frame_state] x 48 (height) x 68 (width) x 3 (channels)
    # Labels: count (in train/test set) x 1 (index of action in ACTIONS array)
    print('Train states shape: ' + str(s_train.shape))
    print('Train actions shape: ' + str(a_train.shape))
    print('Train scores shape: ' + str(scores_train.shape))
    print('Train healths shape: ' + str(h_train.shape))
    print('Test states shape: ' + str(s_test.shape))
    print('Test actions shape: ' + str(a_test.shape))
    print('Test scores shape: ' + str(scores_test.shape))
    print('Test healths shape: ' + str(h_test.shape))

    # Returns
    return (s_train, a_train, scores_train, h_train), (s_test, a_test, scores_test, h_test)

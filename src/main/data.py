# Parcel data into usable format

import os
import re
from sklearn.model_selection import train_test_split
from scipy import ndimage
from tqdm import *

def load_datasets(tier, params):
    # Matching group 1 is frame number, matching group 2 is action
    pattern = re.compile('(\d+)_(.*).png')

    # Load images
    print("Loading images:")
    images = []
    labels = []
    for filename in tqdm(os.listdir(params["data_dir"])):
        # Stop early if have enough images
        if params["num_images"] != -1 and len(images) > params["num_images"]:
            break

        # Extract metadata
        match = re.search(pattern, filename)
        frame_number = match.group(1)
        action = match.group(2)

        # Convert image and add to collection
        img = ndimage.imread(params["data_dir"] + filename)
        if img is not None:
            images.append(img)
            labels.append(action)
    print("Loaded", len(images), "images.")

    # Partition
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=params["eval_proportion"], random_state=42)
    print("Train count: X:", len(X_train), "y:", len(y_train))
    print("Test count: X:", len(X_test), "y:", len(y_test))

    # TODO: A state consists of three consecutive frames

    # Return
    return (X_train, y_train), (X_test, y_test)

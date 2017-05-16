import argparse
import numpy as np
import tensorflow as tf

from data import load_datasets

# COMMAND LINE ARGUMENTS
tf.app.flags.DEFINE_bool("dev", False, "")
tf.app.flags.DEFINE_bool("test", False, "")
tf.app.flags.DEFINE_bool("validate", False, "")
tf.app.flags.DEFINE_integer("num_images", -1, "")
tf.app.flags.DEFINE_float("eval_proportion", 0.1, "")
# tf.app.flags.DEFINE_integer("num_train", 10000, "")
# tf.app.flags.DEFINE_integer("num_dev", 1000, "")
# tf.app.flags.DEFINE_integer("num_test", 1000, "")

# HYPERPARAMETERS
tf.app.flags.DEFINE_float("lr", 0.0004, "Learning rate.")
tf.app.flags.DEFINE_integer("cnn_hidden_size", 300, "Size of each model layer.")

# INFRASTRUCTURE
tf.app.flags.DEFINE_string("data_dir", "../../data/labeled_051517_2114/", "data directory (default ./data)")
tf.app.flags.DEFINE_integer("batch_size", 5, "")

FLAGS = tf.app.flags.FLAGS

def main(_):
    assert(FLAGS.validate or ((FLAGS.dev and not FLAGS.test) or (FLAGS.test and not FLAGS.dev))), "When not validating, must set exaclty one of --dev or --test flag to specify evaluation dataset."

    # Set random seed
    np.random.seed(244)

    # Load the two pertinent datasets into train_dataset and eval_dataset
    data_params = {
        "data_dir": FLAGS.data_dir,
        "num_images": FLAGS.num_images,
        "eval_proportion": FLAGS.eval_proportion,
        "image_size": 28,
        "batch_size": FLAGS.batch_size
    }
    if FLAGS.test:
        train_dataset, eval_dataset = load_datasets('test', data_params)
    else:
        train_dataset, eval_dataset = load_datasets('dev', data_params)

    # Train model
    # todo

if __name__ == "__main__":
    tf.app.run()

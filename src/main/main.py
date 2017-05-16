import argparse
import numpy as np
import tensorflow as tf

# COMMAND LINE ARGUMENTS
tf.app.flags.DEFINE_bool("dev", False, "")
tf.app.flags.DEFINE_bool("test", False, "")
tf.app.flags.DEFINE_bool("validate", False, "")
tf.app.flags.DEFINE_integer("num_train", 10000, "")
tf.app.flags.DEFINE_integer("num_dev", 1000, "")
tf.app.flags.DEFINE_integer("num_test", 1000, "")

# HYPERPARAMETERS
tf.app.flags.DEFINE_float("lr", 0.0004, "Learning rate.")
tf.app.flags.DEFINE_integer("cnn_hidden_size", 300, "Size of each model layer.")

# INFRASTRUCTURE
tf.app.flags.DEFINE_string("data_dir", "data/testdata", "data directory (default ./data)")

FLAGS = tf.app.flags.FLAGS

def main(_):
    assert(FLAGS.validate or ((FLAGS.dev and not FLAGS.test) or (FLAGS.test and not FLAGS.dev))), "When not validating, must set exaclty one of --dev or --test flag to specify evaluation dataset."

    # Set random seed
    np.random.seed(244)

    # Load the two pertinent datasets into train_dataset and eval_dataset
    train_dataset = load_dataset('train', FLAGS.num_train)
    if FLAGS.test:
        eval_dataset = load_dataset('test', FLAGS.num_test)
    else:
        eval_dataset = load_dataset('dev', FLAGS.num_dev)

    # Train model
    # todo

if __name__ == "__main__":
    tf.app.run()

import argparse
import numpy as np
import tensorflow as tf

from data import load_datasets
from foxnet_model import FoxNetModel

# COMMAND LINE ARGUMENTS
tf.app.flags.DEFINE_bool("dev", False, "")
tf.app.flags.DEFINE_bool("test", False, "")
tf.app.flags.DEFINE_string("model", "simple", "")
tf.app.flags.DEFINE_bool("validate", False, "")
tf.app.flags.DEFINE_integer("num_images", -1, "")
tf.app.flags.DEFINE_float("eval_proportion", 0.1, "")
# tf.app.flags.DEFINE_integer("num_train", 10000, "")
# tf.app.flags.DEFINE_integer("num_dev", 1000, "")
# tf.app.flags.DEFINE_integer("num_test", 1000, "")

# HYPERPARAMETERS
tf.app.flags.DEFINE_integer("frames_per_state", 3, "")
tf.app.flags.DEFINE_float("lr", 0.0004, "Learning rate.")
tf.app.flags.DEFINE_integer("cnn_hidden_size", 300, "Size of each model layer.")

# INFRASTRUCTURE
tf.app.flags.DEFINE_string("data_dir", "../../data/", "data directory (default ./data)")
tf.app.flags.DEFINE_integer("image_width", 640, "")
tf.app.flags.DEFINE_integer("image_height", 480, "")
tf.app.flags.DEFINE_integer("num_channels", 3, "")
tf.app.flags.DEFINE_integer("num_actions", 7, "")
tf.app.flags.DEFINE_integer("batch_size", 5, "")

FLAGS = tf.app.flags.FLAGS

def get_data_params():
    return {
        "data_dir": FLAGS.data_dir,
        "num_images": FLAGS.num_images,
        "frames_per_state": FLAGS.frames_per_state,
        "eval_proportion": FLAGS.eval_proportion,
        "image_size": 28,
        "batch_size": FLAGS.batch_size
    }

def initialize_model(session, model):
    logging.info("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def run_model(train_dataset, eval_dataset, lr):
    # Reset every time
    tf.reset_default_graph()
    tf.set_random_seed(1)

    foxnet = FoxNetModel(
                FLAGS.model,
                FLAGS.lr,
                FLAGS.image_height,
                FLAGS.image_width,
                FLAGS.num_channels,
                FLAGS.frames_per_state,
                FLAGS.num_actions
            )

    with tf.Session() as sess:
        initialize_model(sess, foxnet)
        # epoch_number, train_accuracy, train_loss, error = foxnet.train(sess, train_dataset, FLAGS.batch_size)
        # test_accuracy, avg_test_loss = foxnet.evaluate_prediction(sess, eval_dataset, FLAGS.batch_size)
        # return (epoch_number, train_accuracy, train_loss, test_accuracy, avg_test_loss)

def main(_):
    # TODO: Eventually, should have separate dev and test datasets and require that we specify which we want to use.
    # assert(FLAGS.validate or ((FLAGS.dev and not FLAGS.test) or (FLAGS.test and not FLAGS.dev))), "When not validating, must set exaclty one of --dev or --test flag to specify evaluation dataset."

    # Set random seed
    np.random.seed(244)

    # Load the two pertinent datasets into train_dataset and eval_dataset
    if FLAGS.test:
        train_dataset, eval_dataset = load_datasets('test', get_data_params())
    else:
        train_dataset, eval_dataset = load_datasets('dev', get_data_params())

    # Train model
    run_model(train_dataset, eval_dataset, FLAGS.lr)

if __name__ == "__main__":
    tf.app.run()

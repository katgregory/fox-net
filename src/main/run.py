import os
import numpy as np
import tensorflow as tf

from foxnet_model import FoxNetModel
from emu_interact import FrameReader
from data_manager import DataManager
from scipy.misc import imresize


# COMMAND LINE ARGUMENTS
tf.app.flags.DEFINE_bool("dev", False, "")
tf.app.flags.DEFINE_bool("test", False, "")
tf.app.flags.DEFINE_string("model", "fc", "Options: fc, simple_cnn, dqn") 
tf.app.flags.DEFINE_bool("validate", False, "")
tf.app.flags.DEFINE_bool("multi_frame_state", False, "If false, overrides num_frames & reduces dimension of data")
tf.app.flags.DEFINE_integer("num_images", 1000, "")
tf.app.flags.DEFINE_float("eval_proportion", 0.5, "") # TODO: Right now, breaks unless same size as train data
tf.app.flags.DEFINE_bool("validate_incrementally", False, "")
tf.app.flags.DEFINE_bool("plot_losses", True, "")
tf.app.flags.DEFINE_bool("plot_accuracies", True, "")

tf.app.flags.DEFINE_bool("load_model", False, "")
tf.app.flags.DEFINE_bool("save_model", True, "")
tf.app.flags.DEFINE_string("model_dir", "sample_model", "Directory with a saved model's files")
tf.app.flags.DEFINE_bool("train_offline", False, "")
tf.app.flags.DEFINE_bool("train_online", False, "")
tf.app.flags.DEFINE_bool("qlearning", False, "")

tf.app.flags.DEFINE_string("ip", "127.0.0.1", "Specify host IP. Default is local loopback.")
# LAYER SIZES
tf.app.flags.DEFINE_integer("cnn_filter_size", 7, "Size of filter.")
tf.app.flags.DEFINE_integer("cnn_num_filters", 32, "Filter count.")

# HYPERPARAMETERS
tf.app.flags.DEFINE_integer("frames_per_state", 3, "")
tf.app.flags.DEFINE_float("lr", 0.000004, "Learning rate.")
tf.app.flags.DEFINE_integer("num_epochs", 20, "")

# INFRASTRUCTURE
tf.app.flags.DEFINE_string("data_dir", "./data/data_053017/", "data directory (default ./data)")
tf.app.flags.DEFINE_string("results_dir", "./results/", "")
tf.app.flags.DEFINE_integer("image_width", 64, "")
tf.app.flags.DEFINE_integer("image_height", 48, "")
tf.app.flags.DEFINE_integer("num_channels", 3, "")
tf.app.flags.DEFINE_integer("batch_size", 20, "")

ACTIONS = ['w', 'a', 's', 'd', 'j', 'k', 'n']
ACTION_NAMES = ['up', 'left', 'down', 'right', 'fire', 'back', 'do nothing']

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model):
    session.run(tf.global_variables_initializer())
    print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def run_model():
    # Reset every time
    tf.reset_default_graph()
    tf.set_random_seed(1)

    # Get the session.
    session = tf.Session()

    # Initialize a FoxNet model.
    foxnet = FoxNetModel(
                FLAGS.model,
                FLAGS.qlearning,
                FLAGS.lr,
                FLAGS.image_height,
                FLAGS.image_width,
                FLAGS.num_channels,
                FLAGS.multi_frame_state,
                FLAGS.frames_per_state,
                ACTIONS,
                FLAGS.cnn_filter_size,
                FLAGS.cnn_num_filters
            )


    # Initialize a data manager.
    data_manager = DataManager()
    if FLAGS.train_online:
        data_manager.init_online(foxnet, session, FLAGS.batch_size, FLAGS.ip, FLAGS.image_height, FLAGS.image_width, 0.1)
    else:
        data_manager.init_offline(FLAGS.test, get_data_params(), FLAGS.batch_size)

    # Load pretrained model
    if FLAGS.load_model:
        # Create an object to get emulator frames
        frame_reader = FrameReader(FLAGS.ip, FLAGS.image_height, FLAGS.image_width)

        # Load the model
        model_dir = './models/%s' % (FLAGS.model_dir)
        model_name = '%s' % (FLAGS.model_dir)
        print('Loading model from dir: %s' % model_dir)
        sv = tf.train.Supervisor(logdir=model_dir)
        with sv.managed_session() as session:
            if not sv.should_stop():
                if FLAGS.train_online == True:
                    foxnet.run_q_learning(data_manager, session)
    else:
        # Train a new model.
        initialize_model(session, foxnet)
        print('Training...')

        # Run Q-learning or classification.
        if FLAGS.qlearning:
            foxnet.run_q_learning(data_manager, session)
        else:
            foxnet.run_classification(data_manager,
                                      session,
                                      epochs=FLAGS.num_epochs,
                                      training_now=True,
                                      validate_incrementally=FLAGS.validate_incrementally,
                                      print_every=1,
                                      plot_losses=FLAGS.plot_losses,
                                      plot_accuracies=FLAGS.plot_accuracies,
                                      results_dir=FLAGS.results_dir
                                      )

    # Save the model
    if FLAGS.save_model:
        # Save model
        model_dir = './models/%s' % (FLAGS.model_dir)
        model_name = '%s' % (FLAGS.model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        saver = tf.train.Saver()
        saver.save(session, model_dir + '/' + model_name)
        print('Saved model to dir: %s' % model_dir)

    # Validate the model
    # print('Validating...')
    # foxnet.run(sess, X_eval, y_eval, FLAGS.batch_size, epochs=1)

    # Close session
    session.close()

def get_data_params():
    return {
        "data_dir": FLAGS.data_dir,
        "num_images": FLAGS.num_images,
        "width": FLAGS.image_width,
        "height": FLAGS.image_height,
        "multi_frame_state": FLAGS.multi_frame_state,
        "frames_per_state": FLAGS.frames_per_state,
        "actions": ACTIONS,
        "eval_proportion": FLAGS.eval_proportion,
        "image_size": 28,
    }

def main(_):
    # TODO: Eventually, should have separate dev and test datasets and require that we specify which we want to use.
    # assert(FLAGS.validate or ((FLAGS.dev and not FLAGS.test) or (FLAGS.test and not FLAGS.dev))), "When not validating, must set exaclty one of --dev or --test flag to specify evaluation dataset."

    # Set random seed
    np.random.seed(244)

    # Train model
    run_model()

if __name__ == "__main__":
    tf.app.run()

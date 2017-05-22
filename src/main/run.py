import os
import numpy as np
import tensorflow as tf

from data import load_datasets
from foxnet_model import FoxNetModel
from emu_interact import FrameReader
from scipy.misc import imresize


# COMMAND LINE ARGUMENTS
tf.app.flags.DEFINE_bool("dev", False, "")
tf.app.flags.DEFINE_bool("test", False, "")
tf.app.flags.DEFINE_string("model", "fc", "Options: fc, simple_cnn") 
tf.app.flags.DEFINE_bool("validate", False, "")
tf.app.flags.DEFINE_bool("multi_frame_state", False, "If false, overrides num_frames & reduces dimension of data")
tf.app.flags.DEFINE_integer("num_images", 1000, "")
tf.app.flags.DEFINE_float("eval_proportion", 0.5, "") # TODO: Right now, breaks unless same size as train data
tf.app.flags.DEFINE_bool("validate_incrementally", True, "")
tf.app.flags.DEFINE_bool("plot_losses", True, "")
tf.app.flags.DEFINE_bool("plot_accuracies", True, "")

tf.app.flags.DEFINE_bool("load_model", False, "")
tf.app.flags.DEFINE_string("model_dir", "", "Directory with a saved model's files")

# LAYER SIZES
tf.app.flags.DEFINE_integer("cnn_filter_size", 7, "Size of filter.")
tf.app.flags.DEFINE_integer("cnn_num_filters", 32, "Filter count.")

# HYPERPARAMETERS
tf.app.flags.DEFINE_integer("frames_per_state", 3, "")
tf.app.flags.DEFINE_float("lr", 0.000004, "Learning rate.")
tf.app.flags.DEFINE_integer("num_epochs", 20, "")

# INFRASTRUCTURE
tf.app.flags.DEFINE_string("data_dir", "./data/data_051617/", "data directory (default ./data)")
tf.app.flags.DEFINE_string("results_dir", "./results/", "")
tf.app.flags.DEFINE_integer("image_width", 64, "")
tf.app.flags.DEFINE_integer("image_height", 48, "")
tf.app.flags.DEFINE_integer("num_channels", 3, "")
tf.app.flags.DEFINE_integer("batch_size", 20, "")

ACTIONS = ['w', 'a', 's', 'd', 'j', 'l', 'n']

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model):
    session.run(tf.global_variables_initializer())
    print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
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
                FLAGS.multi_frame_state,
                FLAGS.frames_per_state,
                ACTIONS,
                FLAGS.cnn_filter_size,
                FLAGS.cnn_num_filters
            )

    X_train, y_train = train_dataset
    X_eval, y_eval = eval_dataset

    if FLAGS.load_model:
        # Create an object to get emulator frames
        frame_reader = FrameReader()

        # Load the model
        # saver = tf.train.import_meta_graph('./model/saved_model.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        # print(tf.global_variables())

        model_dir = './models/%s' % FLAGS.model_dir
        print('Loading model from dir: %s' % model_dir)
        sv = tf.train.Supervisor(logdir=model_dir, save_model_secs=60)
        with sv.managed_session() as sess:
            if not sv.should_stop():
                while True:
                    X_emu = frame_reader.read_frame()
                    X_emu = imresize(X_emu, (FLAGS.image_height, FLAGS.image_width, 3))
                    X_emu = np.expand_dims(X_emu, axis=0)
                    # pred = foxnet.run(foxnet.probs, feed_dict={foxnet.X:X_emu})
                    pred = sess.run(foxnet.probs, feed_dict={foxnet.X:X_emu, foxnet.is_training:False})
                    action = ACTIONS[np.argmax(pred)]
                    print("action: " + str(action))
                    frame_reader.send_action(action)
    
    else:  
        with tf.Session() as sess:
        # Train a new model

            initialize_model(sess, foxnet)
            print('Training...')
            foxnet.run(
                    sess,
                    X_train,
                    y_train,
                    FLAGS.batch_size,
                    epochs=FLAGS.num_epochs,
                    training_now=True,
                    validate_incrementally=FLAGS.validate_incrementally,
                    X_eval=X_eval,
                    y_eval=y_eval,
                    print_every=1,
                    plot_losses=FLAGS.plot_losses,
                    plot_accuracies=FLAGS.plot_accuracies,
                    results_dir=FLAGS.results_dir
                )

            # Save model
            model_dir = './models/%s' % FLAGS.model_dir
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            saver = tf.train.Saver()
            saver.save(sess, model_dir)
            print('Saved model to dir: %s' % model_dir)

            print('Validating...')
            foxnet.run(sess, X_eval, y_eval, FLAGS.batch_size, epochs=1)

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

    # Load the two pertinent datasets into train_dataset and eval_dataset
    if FLAGS.test:
        train_dataset, eval_dataset = load_datasets('test', get_data_params())
    else:
        train_dataset, eval_dataset = load_datasets('dev', get_data_params())

    # Train model
    run_model(train_dataset, eval_dataset, FLAGS.lr)

if __name__ == "__main__":
    tf.app.run()

import datetime
import json
import numpy as np
import os
import sys
import tensorflow as tf

from foxnet import FoxNet
from data_manager import DataManager


# COMMAND LINE ARGUMENTS
tf.app.flags.DEFINE_bool("dev", False, "")
tf.app.flags.DEFINE_bool("test", False, "")
tf.app.flags.DEFINE_string("model", "fc", "Options: fc, simple_cnn, dqn, dqn_3d")
tf.app.flags.DEFINE_bool("validate", True, "Validate after all training is complete")
tf.app.flags.DEFINE_bool("validate_incrementally", False, "Validate after every epoch")
tf.app.flags.DEFINE_integer("num_images", 1000, "")
tf.app.flags.DEFINE_float("eval_proportion", 0.2, "") # TODO: Right now, breaks unless same size as train data
tf.app.flags.DEFINE_bool("plot", True, "")
tf.app.flags.DEFINE_bool("verbose", False, "")

# MODEL SAVING
tf.app.flags.DEFINE_bool("load_model", False, "")
tf.app.flags.DEFINE_string("load_model_dir", "load_model", "Directory with a saved model's files.")
tf.app.flags.DEFINE_bool("save_model", True, "")
tf.app.flags.DEFINE_string("save_model_dir", "sample_model", "Directory in which to save this model's files.")

# TRAINING
tf.app.flags.DEFINE_bool("train_offline", False, "")
tf.app.flags.DEFINE_bool("train_online", False, "")
tf.app.flags.DEFINE_integer("max_batches", -1, "")
tf.app.flags.DEFINE_bool("qlearning", False, "")
tf.app.flags.DEFINE_bool("user_overwrite", False, "")
tf.app.flags.DEFINE_string("ip", "127.0.0.1", "Specify host IP. Default is local loopback.")

# LAYER SIZES
tf.app.flags.DEFINE_integer("cnn_filter_size", 7, "Size of filter.")
tf.app.flags.DEFINE_integer("cnn_num_filters", 32, "Filter count.")

# HYPERPARAMETERS
tf.app.flags.DEFINE_integer("frames_per_state", 1, "")
tf.app.flags.DEFINE_float("lr", 0.000004, "Learning rate.")
tf.app.flags.DEFINE_float("reg_lambda", .01, "Regularization")
tf.app.flags.DEFINE_integer("num_epochs", 20, "")
tf.app.flags.DEFINE_float("epsilon", 0.05, "E-greedy exploration rate.")
tf.app.flags.DEFINE_float("health_weight", 10.0, "Amount to weight health reward.")

# TARGET NETWORK
tf.app.flags.DEFINE_bool("use_target_net", True, "")
tf.app.flags.DEFINE_integer("target_q_update_freq", 100, "")

# INFRASTRUCTURE
tf.app.flags.DEFINE_string("data_dir", "./data/data_053017/", "data directory (default ./data)")
tf.app.flags.DEFINE_string("results_dir", "./results/", "")
tf.app.flags.DEFINE_integer("image_width", 64, "")
tf.app.flags.DEFINE_integer("image_height", 48, "")
tf.app.flags.DEFINE_integer("num_channels", 3, "")
tf.app.flags.DEFINE_integer("batch_size", 10, "")
tf.app.flags.DEFINE_integer("replay_buffer_size", 1000, "")

ACTIONS = ['w', 'a', 's', 'd', 'j', 'k', 'n']
ACTION_NAMES = ['up', 'left', 'down', 'right', 'fire', 'back', 'do nothing']

FLAGS = tf.app.flags.FLAGS

def record_params():
    dt = str(datetime.datetime.now())
    # Record params
    f = open(FLAGS.results_dir + "params" + "/" + dt + ".txt","w+")
    f.write(" ".join(sys.argv) + "\n\n")
    for flag in FLAGS.__flags:
        f.write(flag + ":" + str(FLAGS.__flags[flag]) + "\n")
    f.close()
    # Dump flags in case we want to load this model later
    saving_flags_file = FLAGS.results_dir + "flags" + "/" + FLAGS.save_model_dir + "_" + dt + ".json"
    with open(saving_flags_file, "w+") as f:
        json.dump(FLAGS.__flags, f)
    return dt

def initialize_model(session, model):
    print("##### MODEL ###############################################")
    session.run(tf.global_variables_initializer())
    print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    print("Flags: " + str(FLAGS.__flags))
    return model

def get_model_path(model_name):
    model_dir = './models/%s' % model_name
    model_path = model_dir + '/' + model_name
    return model_dir, model_path

def construct_model_with_flags(session, flags, load_model, load_model_dir=None, load_model_path=None, save_model_path=None):
    return FoxNet(
                session,
                flags['model'],
                flags['qlearning'],
                flags['lr'],
                flags['reg_lambda'],
                flags['use_target_net'],
                flags['target_q_update_freq'],
                flags['image_height'],
                flags['image_width'],
                flags['num_channels'],
                flags['frames_per_state'],
                ACTIONS,
                ACTION_NAMES,
                flags['cnn_filter_size'],
                flags['cnn_num_filters'],
                load_model,
                load_model_dir,
                load_model_path,
                flags['save_model'],
                save_model_path
            )

def load_flags(results_dir, load_model_dir):
    loading_flags_files = [filename for filename in os.listdir(results_dir + "flags") if filename.startswith(load_model_dir + "_")]
    if len(loading_flags_files) == 0:
        print("Uh oh! Can't find flag file for model to load. Remember, should be ./results/flags/[load_model_dir name]_[timestamp]")
        assert(0)
    loading_flag_file = results_dir + "flags" + "/" + loading_flags_files[-1]
    with open(loading_flag_file, 'r') as f:
        model_flags = json.load(f)
    return model_flags

def get_model():
    # Reset every time
    tf.reset_default_graph()
    tf.set_random_seed(1)

    # Get the session.
    session = tf.Session()

    # Construct relevant file names and paths for loading / saving models
    if (FLAGS.load_model):
        load_model_dir, load_model_path = get_model_path(FLAGS.load_model_dir)
    else:
        load_model_dir, load_model_path = None, None
    save_model_dir, save_model_path = get_model_path(FLAGS.save_model_dir)

    # Get model flags
    if FLAGS.load_model:
        model_flags = load_flags(FLAGS.results_dir, FLAGS.load_model_dir)
    else:
        model_flags = FLAGS.__flags

    # Initialize a FoxNet model.
    foxnet = construct_model_with_flags(session, model_flags, FLAGS.load_model, load_model_dir, load_model_path, save_model_path)

    # Set up saving
    if FLAGS.save_model and not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # Initialize model's global variables
    initialize_model(session, foxnet)

    return session, foxnet

# Returns a configured data manager
def initialize_data_manager(foxnet, session):
    data_manager = DataManager(FLAGS.verbose)
    if FLAGS.train_online:
        frames_per_state = 1
        if FLAGS.model == "dqn_3d":
            frames_per_state = FLAGS.frames_per_state
        data_manager.init_online(foxnet, session, FLAGS.batch_size, FLAGS.replay_buffer_size, frames_per_state,
                                 FLAGS.ip, FLAGS.image_height, FLAGS.image_width, FLAGS.epsilon, FLAGS.health_weight,
                                 FLAGS.user_overwrite)
    else:
        data_manager.init_offline(FLAGS.test, get_data_params(), FLAGS.batch_size)
    return data_manager

def run_model():
    # Starting a new training session!
    dt = record_params()
    print("Session timestamp: " + dt)

    # Either load the specified model or create a new one with the given flags
    session, foxnet = get_model()

    # Initialize a data manager
    data_manager = initialize_data_manager(foxnet, session)

    # Train the model, using Q-learning or classification.
    print("##### TRAINING ############################################")
    if FLAGS.qlearning:
        foxnet.run_q_learning(data_manager, 
                                session, 
                                FLAGS.num_epochs, 
                                max_batches=FLAGS.max_batches,
                                results_dir=FLAGS.results_dir,
                                plot=FLAGS.plot,
                                dt=dt
                                )
    else:
        # Classification no longer works on this branch
        foxnet.run_classification(data_manager,
                                  session,
                                  epochs=FLAGS.num_epochs,
                                  training_now=True,
                                  validate_incrementally=FLAGS.validate_incrementally,
                                  print_every=1,
                                  plot=FLAGS.plot,
                                  results_dir=FLAGS.results_dir,
                                  dt=dt
                                  )

    # Save the model
    foxnet.save(session)

    # Validate the model
    if (FLAGS.validate and not FLAGS.train_online and not FLAGS.qlearning):
        print("##### VALIDATING ##########################################")
        foxnet.run_validation(data_manager, session)

    # Close session
    session.close()

def get_data_params():
    return {
        "data_dir": FLAGS.data_dir,
        "num_images": FLAGS.num_images,
        "width": FLAGS.image_width,
        "height": FLAGS.image_height,
        "multi_frame_state": FLAGS.model == "dqn_3d",
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

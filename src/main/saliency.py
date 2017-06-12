# Based on Kat's CS231N Assignment 3
# From from foxnet/ as python src/main/saliency.py

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from data_manager import DataManager
from foxnet_model import FoxNetModel
from run import get_model_path
from run import construct_model_with_flags
from run import load_flags

ACTIONS = ['w', 'a', 's', 'd', 'j', 'k', 'l', 'n']
ACTION_NAMES = ['up', 'left', 'down', 'right', 'fire', 'back', 'start', 'do nothing']

model_to_load = "sample_model"#'kevin_highscoring_model'
results_dir = './results/'

def compute_saliency_maps(sess, s, a, r, sp, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A FoxNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.

    correct_scores = tf.gather_nd(model.q_values,
    correct_scores = tf.gather_nd(model.model.get_q_values_op(model.states),
                                  tf.stack((tf.range(s.shape[0]), tf.cast(model.actions, tf.int32)), axis=1))

    # Use the correct_scores to compute the loss
    total_loss = tf.losses.softmax_cross_entropy(a, logits = correct_scores)
    mean_loss = tf.reduce_mean(total_loss)

    # Use tf.gradients to compute the gradient of the loss with respect to the input image stored in model.image.
    gradients = tf.gradients(correct_scores, model.states)

    # Use the global sess variable to finally run the computation.
    feed_dict = {model.states: s,
                 model.actions: a,
                 model.rewards: r,
                 model.states_p: sp,
                 model.is_training: False }
    _, saliency = sess.run([mean_loss, gradients], feed_dict)

    # Take the absolute value of this gradient, then take the maximum value over the 3 input channels;
    # the final saliency map thus has shape (H, W) and all entries are nonnegative.
    saliency = np.absolute(saliency[0])
    saliency = np.amax(saliency, axis=3)

    return saliency

# show_saliency_maps(X, y, 5)
def show_saliency_maps(sess, model, s, a, r, s_p, count=5, num_trials=5):
    for trial in xrange(num_trials):
        mask = np.random.randint(0, s.shape[0], count)
        sm = s[mask]
        am = a[mask]
        rm = r[mask]
        spm = s_p[mask]

        saliency = compute_saliency_maps(sess, sm, am, rm, spm, model)

        for i in range(mask.size):
            plt.subplot(2, mask.size, i + 1)
            plt.imshow(Xm[i])
            plt.axis('off')
            plt.title(ACTION_NAMES[ym[i]])
            plt.subplot(2, mask.size, mask.size + i + 1)
            plt.title(mask[i])
            plt.imshow(saliency[i], cmap=plt.cm.hot)
            plt.axis('off')
            plt.gcf().set_size_inches(10, 4)
        plt.savefig(results_dir + "saliency/saliency" + str(trial) + ".png")
        plt.close()

def get_data_params():
    return {
        "data_dir": './data/data_053017/',
        "num_images": 100,
        "width": 64,
        "height": 48,
        "multi_frame_state": False,
        "frames_per_state": 1,
        "actions": ACTIONS,
        "eval_proportion": .1,
        "image_size": 28,
    }


def main(_):
    # Load model
    print("##### LOADING MODEL #######################################")
    load_model_dir, load_model_path = get_model_path(model_to_load)
    print('From dir: %s' % load_model_dir)

    tf.reset_default_graph()
    sess = tf.Session()

    flags = load_flags(results_dir, model_to_load)
    flags['save_model'] = False
    foxnet = construct_model_with_flags(sess, flags, True, load_model_dir, load_model_path, "")

    sess.run(tf.global_variables_initializer())

    data_manager = DataManager(False)
    data_manager.init_offline(False, get_data_params(), 10)
    data_manager.init_epoch()
    s, a, r, s_p, _ = data_manager.get_next_batch()

    print("##### SALIENCY MAPS #######################################")
    # Generate 5 options
    show_saliency_maps(sess, foxnet, s, a, r, s_p)

if __name__ == "__main__":
    tf.app.run()

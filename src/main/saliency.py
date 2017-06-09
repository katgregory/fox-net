# Based on Kat's CS231N Assignment 3
# From from foxnet/ as python src/main/saliency.py

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from data import load_datasets
from foxnet_model import FoxNetModel

model_dir = './models/trained_cnn/'
model_name = 'sample_model'
results_dir = './results/saliency/'
ACTIONS = ['w', 'a', 's', 'd', 'j', 'k', 'l', 'n']
ACTION_NAMES = ['up', 'left', 'down', 'right', 'fire', 'back', 'start', 'do nothing']

def compute_saliency_maps(X, y, model):
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

    correct_scores = tf.gather_nd(model.probs,
                                  tf.stack((tf.range(X.shape[0]), tf.cast(model.y, tf.int32)), axis=1))

    # Use the correct_scores to compute the loss
    total_loss = tf.losses.softmax_cross_entropy(y, logits = correct_scores)
    mean_loss = tf.reduce_mean(total_loss)

    # Use tf.gradients to compute the gradient of the loss with respect to the input image stored in model.image.
    gradients = tf.gradients(correct_scores, model.X)

    # Use the global sess variable to finally run the computation.
    feed_dict = {model.X: X,
                 model.y: y,
                 model.is_training: False }
    _, saliency = sess.run([mean_loss, gradients], feed_dict)

    # Take the absolute value of this gradient, then take the maximum value over the 3 input channels;
    # the final saliency map thus has shape (H, W) and all entries are nonnegative.
    saliency = np.absolute(saliency[0])
    saliency = np.amax(saliency, axis=3)

    return saliency

# show_saliency_maps(X, y, 5)
def show_saliency_maps(model, X, y, count=5, num_trials=5):
    for trial in xrange(num_trials):
        mask = np.random.randint(0, X.shape[0], count)
        Xm = X[mask]
        ym = y[mask]

        saliency = compute_saliency_maps(Xm, ym, model)

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
        plt.savefig(results_dir + "saliency" + str(trial) + ".png")
        plt.close()

# Load model
print("##### LOADING MODEL #######################################")
print('From dir: %s' % model_dir)
tf.reset_default_graph()
sess=tf.Session()
saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
saver.restore(sess, tf.train.latest_checkpoint(model_dir))

foxnet = FoxNetModel(
                model='simple_cnn',
                q_learning='False',
                lr=0.000004,
                reg_lambda=-1,
                dropout=0.5
                use_target_net=False,
                tau=.99,
                target_q_update_step=10,
                height=48,
                width=64,
                n_channels=3,
                frames_per_state=1,
                available_actions=ACTIONS,
                available_actions_names=ACTION_NAMES,
                cnn_filter_size=7,
                cnn_n_filters=32
            )

sess.run(tf.global_variables_initializer())

# graph = tf.get_default_graph()
# for var in tf.global_variables():
#     print(var)
# print([var for var in tf.all_variables()])
# graph.get_tensor_by_name('X')

# Load data
def get_data_params():
    return {
        "data_dir": './data/data_053017/',
        "num_images": 1000,
        "width": 64,
        "height": 48,
        "multi_frame_state": False,
        "frames_per_state": 1,
        "actions": ACTIONS,
        "eval_proportion": .5,
        "image_size": 28,
    }

all_data, _ = load_datasets("test", get_data_params())
s, a, scores, h = all_data

print("##### SALIENCY MAPS #######################################")
# Generate 5 options
show_saliency_maps(foxnet, s, a)

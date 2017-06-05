# Based on Kat's CS231N Assignment 3

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
    correct_scores = tf.gather_nd(model.classifier,
                                  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))

    # Use the correct_scores to compute the loss
    total_loss = tf.losses.softmax_cross_entropy(y, logits=correct_scores)
    mean_loss = tf.reduce_mean(total_loss)
        
    # Use tf.gradients to compute the gradient of the loss with respect to the input image stored in model.image. 
    gradients = tf.gradients(correct_scores, model.image)
    
    # Use the global sess variable to finally run the computation.
    feed_dict = {model.image: X,
                 model.labels: y }
    _, saliency = sess.run([mean_loss, gradients], feed_dict)
    
    # Take the absolute value of this gradient, then take the maximum value over the 3 input channels; 
    # the final saliency map thus has shape (H, W) and all entries are nonnegative.
    saliency = np.absolute(saliency[0])
    saliency = np.amax(saliency, axis=3)

    return saliency

# show_saliency_maps(X, y, 5)
def show_saliency_maps(X, y, count):
    mask = np.asarray(np.arrange(count))
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

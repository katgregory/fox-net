import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 
        ##### VARIABLES #####
        with tf.variable_scope(scope, reuse=reuse):
            Wconv1 = tf.get_variable("Wconv1", shape=[8, 8, 4, 32])
            bconv1 = tf.get_variable("bconv1", shape=[32])

            Wconv2 = tf.get_variable("Wconv2", shape=[4, 4, 32, 64])
            bconv2 = tf.get_variable("bconv2", shape=[64])

            Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 64, 64])
            bconv3 = tf.get_variable("bconv3", shape=[64])

            W1 = tf.get_variable("W1", shape=[6400, 512])
            b1 = tf.get_variable("b1", shape=[512])

            W2 = tf.get_variable("W2", shape=[512,num_actions])
            b2 = tf.get_variable("b2", shape=[num_actions])
        ##### GRAPH #####
        # First hidden layer
        # 32 filters of 8x8 with stride 4, then relu
        out = tf.nn.conv2d(out, Wconv1, strides=[1,4,4,1], padding='SAME')+bconv1
        out = tf.nn.relu(out)
        # Second hidden layer
        # 64 filters of 4x4 with stride 2, then relu
        out = tf.nn.conv2d(out, Wconv2, strides=[1,2,2,1], padding='SAME')+bconv2
        out = tf.nn.relu(out)
        # Third hidden layer
        # 64 filters of 3x3 with stride 1, then relu
        out = tf.nn.conv2d(out, Wconv3, strides=[1,1,1,1], padding='SAME')+bconv3
        out = tf.nn.relu(out)
        # Fully-connected with 512 rectifier units
        out = tf.reshape(out, [-1, 6400])
        out = tf.matmul(out, W1) + b1
        # Fully-connected with output 4 units
        out = tf.matmul(out, W2) + b2
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)

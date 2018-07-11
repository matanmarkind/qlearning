from BaseDoubleQnet import BaseDoubleQnet
import tensorflow as tf

class GridworldQnet(BaseDoubleQnet):
    def __init__(self, img_shape, n_actions, future_discount, learning_rate,
                 experiences_size, batch_size, update_target_net_rate,
                 optimizer):
        """
        Hopefully this should work for grayscale as well as RGB.
        :param img_shape: The shape of the image which will be the input to the
        NN.
        :param n_actions: Number of possible actions the system can take.
        :param future_discount: How much less valuable future rewards are
            than current rewards. (gamma)
        :param experiences_size: s=number of past experiences to hold for replay
        :param batch_size: number of experiences to replay at a time
        :param update_target_net_rate: Number of updates to do before copying over
            the main_net to the action_net. (tau)
        :param optimizer: tf.train.<SomeOptimizer>
        """
        BaseDoubleQnet.__init__(
            self, img_shape, n_actions, future_discount,learning_rate,
            experiences_size, batch_size, update_target_net_rate,
            optimizer)

    def make_conv_nn(self):
        """
        This is a method that each specific Q Network will implement based on
        the game they are playing. The output will be split 50/50 along the
        filters for use in a deuling Net, so the result should have an even
        number of filters.
        :return: The final layer of the convolutional NN section to the QNet.
        """
        initializer = tf.contrib.layers.xavier_initializer_conv2d
        conv1 = tf.layers.conv2d(
            inputs=self.state_input, filters=32, kernel_size=(8, 8),
            strides=(4, 4), padding='valid', activation=tf.nn.relu,
            kernel_initializer=initializer(), bias_initializer=initializer())
        conv2 = tf.layers.conv2d(
            inputs=conv1, filters=64, kernel_size=(4 ,4), strides=(2, 2),
            padding='valid', activation=tf.nn.relu,
            kernel_initializer=initializer(), bias_initializer=initializer())
        conv3 = tf.layers.conv2d(
            inputs=conv2, filters=64, kernel_size=(3, 3), strides=(1, 1),
            padding='valid', activation=tf.nn.relu,
            kernel_initializer=initializer(), bias_initializer=initializer())
        conv4 = tf.layers.conv2d(
            inputs=conv3, filters=512, kernel_size=(7, 7), strides=(1, 1),
            padding='valid', activation=tf.nn.relu,
            kernel_initializer=initializer(), bias_initializer=initializer())
        return conv4

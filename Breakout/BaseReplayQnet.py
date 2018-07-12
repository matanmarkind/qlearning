from abc import ABCMeta, abstractmethod
from ExperienceBuffer import ExpBuf
import tensorflow as tf
import numpy as np


class BaseReplayQnet(metaclass=ABCMeta):
    """
    This class is meant to work as a base for doing Q learning with
    experience replay. Doesn't provide a ton of functionality, but helps
    provide an outline for writing a Qnet.

    Designed for tensorflow.
    """
    def __init__(self, input_shape, n_actions, batch_size,
                 optimizer, exp_buf_capacity, exp_buf = ExpBuf, discount = .99,
                 initializer = tf.contrib.layers.xavier_initializer):
        """
        Initializes the base with the values we expect all Qnet's will want. To
        save some boilerplate later on.
        :param input_shape: Iterable describing the shape of the state to be
        input to the NN.
        :param n_actions:
        :param batch_size:
        :param exp_buf_capacity: Number of old states to hold onto for experience
        replay.
        :param exp_buf: Which experience buffer to use. Must provide the
        following interface:
        - Constructor(capacity)
        - append(state, action, reward, next_state, is_terminal)
        - sample(n) - returns lists of len n of
            (state, action, reward, next_state, not_terminal)
        - len(exp_buf) = number of experiences
        :param optimizer: tf.train.<optimizer>()
        :param discount: Discount of future rewards.
        :param initializer: Initializer for weights and biases of network(s).
        """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.discount = discount
        self.initializer = initializer
        self.exp_buf = exp_buf(exp_buf_capacity)

        # Create a tensor to take batches of inputs
        shape = [None]
        for i in input_shape:
            shape.append(i)
        self.state_input = tf.placeholder(shape = shape, dtype=tf.float32)

        # Create the NN used to predict the value of each action for a given
        # state. Called main_net since this network is the main focus of what
        # we are doing. Also can be helpful later if want to have other
        # networks (e.g. DoubleDQN).
        self.main_net = self.make_nn('main_net')

        # Predict the actions that the main_net would take for a batch of
        # states.
        self.prediction = tf.argmax(self.main_net, 1)

        # Create a training operation, and needed inputs, to learn from batches
        # of experiences using some form of SGD.
        # - taken_actions_input - action taken by the network the first time
        #   this state was encountered. From exp_buf. tf.placeholder.
        # - target_vals_input - "true" value of a given state. This is used
        #   as the expected value for calculating the error/loss of the model.
        # - train_op - optimizer.minimize(self.loss). SGD function that the
        #   model will use to update itself.
        res = self.make_train_op(optimizer)
        self.taken_actions_input = res[0]
        self.target_vals_input = res[1]
        self.train_op = res[2]

    def make_nn(self, scope: str):
        """
        Creates a scope for creating the model, which will be implemented by
        the child class.
        :param scope: tf.variable_scope for creating this network within.
        :return: A neural network from self.state_input -> self.n_actions
        """
        with tf.variable_scope(scope):
            return self.make_nn_impl()

    @abstractmethod
    def make_nn_impl(self):
        """
        This is the actual function that creates whatever NN is desired by the
        specifics of this case.
        :return: NN from state_input -> n_actions
        """
        pass

    def make_train_op(self, optimizer):
        """
        Define a training operation so that the main_net can learn. Expects to
        work on batches of experiences from self.exp_buf.
        :param optimizer: tf.train.<optimizer>
        :return: taken_actions_input, target_vals_inpt, train_op
        """
        with tf.variable_scope('train_op'):
            # Create a one hot encoding of the actions taken when this state was
            # first experienced. Used to isolate the current Q value for the
            # action previously taken.
            actions_taken_input = tf.placeholder(shape=(None), dtype=tf.int32)
            actions_onehot = tf.one_hot(actions_taken_input, self.n_actions,
                                        dtype=tf.float32)
            actual_vals = tf.reduce_sum(self.main_net * actions_onehot, axis=1)

            # the "true" Q value.
            target_vals_input = tf.placeholder(shape=(None), dtype=tf.float32)

            # Train by minimizing the loss defined by the full implementation
            # via the SGD optimizer provided by the implementation.
            train_op = optimizer.minimize(
                self.loss(target_vals_input, actual_vals))

            return actions_taken_input, target_vals_input, train_op

    @abstractmethod
    def loss(self, expected, actual):
        """
        A function for calculating the loss of the neural network. Common
        examples include RootMeanSquare or HuberLoss.
        :param expected: a batch of target_vals
        :param actual: a batch of ouputs from the network
        :return: a batch of losses.
        """
        pass


    def rand_action(self):
        """
        Select a random action to perform. Used in preference to
        env.action_space.sample() since there are times when the action_space
        includes useless values like for Breakout.
        :return: [0, n_actions)
        """
        return np.random.randint(0, self.n_actions)

    def add_experience(self, state, action, reward, next_state, is_terminal):
        """
        Add an experience to the expereince buffer.
        """
        self.exp_buf.append(state, action, reward, next_state, is_terminal)

    def predict(self, sess, states):
        """
        Take in a batch of states and predict which action to take for each one.
        :param sess: tf.Session()
        :param states: np.array of states
        :return: np.array of actions to take for each state
        """
        return sess.run(self.prediction,
                        feed_dict={self.state_input: states})

    def exp_buf_size(self):
        """
        How many experiences are held in the experience buffer.
        :return:
        """
        return len(self.exp_buf)

    def exp_buf_capacity(self):
        """
        How many experiences the buffer can hold
        :return:
        """
        return self.exp_buf.capacity

    @abstractmethod
    def update(self, sess):
        """
        An algorithm for using experience replay to teach the model based on
        past experiences.
        :param sess:
        """

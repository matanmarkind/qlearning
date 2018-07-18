from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np

from .ExperienceBuffer import ExpBuf

class BaseReplayQnet(metaclass=ABCMeta):
    """
    This class is meant to work as a base for doing Q learning with
    experience replay. Doesn't provide a ton of functionality, but helps
    provide an outline for writing a Qnet.

    Designed for tensorflow.
    """
    def __init__(self, input_shape, n_actions, batch_size,
                 optimizer, exp_buf_capacity, exp_buf = ExpBuf, discount = .99):
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
        """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.discount = discount
        self.exp_buf = exp_buf(exp_buf_capacity)

        # Create tensors to take batches of inputs
        self.create_inputs()

        # Create the NN used to predict the value of each action for a given
        # state. Called main_net since this network is the main focus of what
        # we are doing. Also can be helpful later if want to have other
        # networks (e.g. DoubleDQN).
        self.main_net = self.make_nn('main_net')

        # Predict the actions that the main_net would take for a batch of
        # states.
        self.prediction = tf.argmax(self.main_net, 1)

        # When an experience is replayed we calculate the predicted value of
        # retaking an action in a past state (state_input, action_input).
        # The loss is then based on comparing this value to some "true" value
        # of having retaken that same action in the past state.
        #
        # The network is then updated based on backpropogating this value.
        self.loss, self.train_op = self.make_train_op(optimizer)

    def create_inputs(self):
      """
      Create tensors to take batches of inputs
      - state = 4 (105, 80) images stacked together.
      - action = used to replay an action taken before.
      - target_vals = "true" Q value for calculating loss.
      """
      state_shape = [None]
      for i in self.input_shape:
          state_shape.append(i)
      self.state_input = tf.placeholder(shape=state_shape, dtype=tf.float32)

      self.action_input = tf.placeholder(shape=(None), dtype=tf.int32)

      self.target_vals_input = tf.placeholder(shape=(None), dtype=tf.float32)

    def make_nn(self, scope: str):
        """
        Creates a scope for creating the model, which will be implemented by
        the child class.
        :param scope: tf.variable_scope for creating this network within.
        :return: A neural network from self.state_input -> self.n_actions
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
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
        Whenever an experience is replayed we calculate how much value the
        network believes it can get if it was back in that old state and
        selected the same action, pred_vals.

        We then calculate the "true" value of being back in state and acting
        the same, target_vals. By comparing the 2 we can calculate the loss
        and use that the learn.

        - action_input - action taken by the network the first time
          this state was encountered. From exp_buf. tf.placeholder.
        - target_vals_input - "true" value of a given state. This is used
          as the expected value for calculating the error/loss of the model.
        - loss - some function that calculates the loss as a function of
          the "true" value of retaking the same action in a past state,
          versus the value the network currently predicts from retaking that
          action back in that past state.
        - train_op - optimizer.minimize(loss). SGD function that the model
          will use to update itself.
          """

        with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
            # Create a one hot encoding of the actions taken when this state was
            # first experienced. Used to isolate the current Q value for the
            # action previously taken.
            actions_onehot = tf.one_hot(self.action_input, self.n_actions,
                                        dtype=tf.float32)
            pred_vals = tf.reduce_sum(self.main_net * actions_onehot, axis=1)

            # Calculate the loss as a function of the target_val versus the
            # predicted value.
            loss = self.loss_fn(self.target_vals_input, pred_vals)

            # Train the network via SGB based on the loss
            train_op = optimizer.minimize(loss)

            return loss, train_op

    @abstractmethod
    def loss_fn(self, expected, actual):
        """
        A function for calculating the loss of the neural network. Common
        examples include RootMeanSquare or HuberLoss.
        :param expected: a batch of target_vals
        :param actual: a batch of outputs from the network
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

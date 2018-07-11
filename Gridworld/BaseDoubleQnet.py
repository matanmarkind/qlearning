from abc import ABCMeta, abstractmethod
from collections import namedtuple, deque
import tensorflow as tf
import numpy as np
import random


class BaseDoubleQnet(metaclass=ABCMeta):
    """
    This class is meant to work as a base for doing Q learning on images.
    This way we don't have to repeat the same boilerplate code every time.
    - Convolution layer - defined per child class by user per game
    - Deuling Networks
    - Experience Replay
    - Double DQN
    """
    # Type used for storing experiences for replay
    ExperienceBuffer = namedtuple(
        'Experience',
        ['state', 'action', 'reward', 'next_state', 'done'])

    def __init__(self, img_shape, n_actions, future_discount, learning_rate,
                 experiences_size, batch_size, update_target_net_rate,
                 optimizer):
        """
        Hopefully this should work for grayscale as well as RGB.
        :param img_shape: The shape of the image which will be the input to the
        NN.
        :param n_actions: Number of possible actions the system can take.
        :param experiences_size: s=number of past experiences to hold for replay
        :param batch_size: number of experiences to replay at a time
        :param update_target_net_rate: Number of updates to do before copying over
            the main_net to the action_net
        :param optimizer: tf.train.<SomeOptimizer>
        """
        self.n_actions = n_actions
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.future_discount = future_discount
        self.update_target_net_rate = update_target_net_rate
        self.update_count = 0 # counter for updating target net
        self.learning_rate = learning_rate
        self.experiences = self.create_experiences(experiences_size)
        self.state_input = tf.placeholder(
            shape=(None, img_shape[0], img_shape[1], img_shape[2]),
            dtype=tf.float32)

        # This NN is used for selecting which action will take place at a given
        # step & is updated every training step.
        self.main_net = self.make_nn('main_net')

        # This NN selects evaluates the value of the action selected by the
        # main_net. This is the secondary net that will be updated by copying
        # from the expector_net every tau updates (tau = update_target_net)
        self.target_net = self.make_nn('target_net')

        # Create a training operation to stochastically learn from batches of
        # experience replay.
        res = self.make_train_op(optimizer)
        self.taken_actions_input = res[0]
        self.target_vals_input = res[1]
        self.train_op = res[2]

        # Batch of predicted moves for a watch of input states.
        self.prediction = tf.argmax(self.main_net, 1)

    @abstractmethod
    def make_conv_nn(self):
        """
        This is a method that each specific Q Network will implement based on
        the game they are playing. The output will be split 50/50 along the
        filters for use in a double Net, so the result should have an even
        number of filters.
        :return: The final layer of the convolutional NN section to the QNet.
        """
        pass

    def make_nn(self, scope: str):
        """
        Splits the final convolutional layer so that it feeds into a deuling
        Qnet. This allows the network to learn the value of being in a given
        state, and separately to learn which action taken in that state will
        be the most advantageous.
        """
        initializer = tf.contrib.layers.xavier_initializer_conv2d
        with tf.variable_scope(scope):
            conv = self.make_conv_nn()  # abstract, set for each game.
            v, a = tf.split(conv, 2, 3)
            value = tf.layers.dense(
                inputs=tf.contrib.layers.flatten(v), units=1,
                kernel_initializer=initializer(),
                bias_initializer=initializer())
            advantage = tf.layers.dense(
                inputs=tf.contrib.layers.flatten(a), units=self.n_actions,
                kernel_initializer=initializer(),
                bias_initializer=initializer())

            # advantage is used to determine the relative advantage of any move
            # value is the base value of just being in this state
            adv_offset = tf.reduce_mean(advantage, axis=1, keepdims=True)
            return value + advantage - adv_offset

    def make_train_op(self, optimizer):
        """
        Define a training operation so that the main_net can learn. Expecting
        to work on batches of experiences from replay.
        - replay the old state by running it through main_net
        - calculate the value of taking the same action as used during
          interactive play. (as opposed to based on the new Q values)
        - get the target_value, calculated in update via doubleQ
        - perform gradient descent to train main_net
        :return: onehot encoded actions, vector of target valus, training_op
        """
        with tf.variable_scope('train_op'):
            # Create a one_hot encoding of the actions taken as a way to
            # isolate the correct Q value. Uses the actions taken by the
            # net previously, to select the Q value.
            actions_taken = tf.placeholder(shape=[None], dtype=tf.int32)
            actions_onehot = tf.one_hot(actions_taken, self.n_actions,
                                        dtype=tf.float32)
            # The "true" Q value. (y*Q(s', a') + r)
            target_vals = tf.placeholder(shape=[None], dtype=tf.float32)
            # Get a batch of values for actions taken by the net previously.
            val = tf.reduce_sum(self.main_net * actions_onehot, axis=1)

            # Train by minimizing the MSE via Gradient Descent
            loss = tf.reduce_mean(tf.square(target_vals - val))
            train_op = optimizer(self.learning_rate).minimize(loss)
            return actions_taken, target_vals, train_op

    def rand_action(self):
        return np.random.randint(0, self.n_actions)

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adds a new experience to be used in replay for training. If the
        experience buffer is full, the deque will automatically pop off the
        oldest experience.
        """
        self.experiences.state.append(state)
        self.experiences.action.append(action)
        self.experiences.reward.append(reward)
        self.experiences.next_state.append(next_state)
        self.experiences.done.append(done)

    # TODO: remove do_print
    def predict(self, sess, states, do_print=False):
        """
        Takes a state and gives the action predicted for that state by the
        main NN
        :param states:
        :return: int in [0, n_actions)
        """
        if do_print:
            # TODO: remove
            print('predict=\n',
                  sess.run(self.main_net,
                           feed_dict={self.state_input: states}))
        return sess.run(self.prediction,
                        feed_dict={self.state_input: states})

    def update(self, sess):
        """
        Updates the Networks based on sampling from past experiences, using
        the Double-DQN algorithm.

        Explain Experience Replay

        Double-DQN: This is a method of training since plain Q learning tends
        to overestimate some actions, but doesn't do so uniformly. This
        separates action selection (main_net) and quality estimation.
        We will then train main_net, while leaving
        target_net constant. Every tau (update_target_net) updates,
        we will copy the main_net over to the target_net so that quality
        estimation will become based on the new learned weights.
        This transforms the Q equation from
        target_val(s, a) = r + y*Q(s', a')
        to
        target_val(s, a) = r + y*target_net(s', argmax(main_net(s', a)))

        :return:
        :param sess: tensorflow session
        """
        self.update_count += self.batch_size
        self.new_sampling()

        next_states = self.sample_next_states()
        # Predict actions for a set of states using main_net.
        actions = self.predict(sess, next_states, True)
        print('actions=', actions)
        # Get a set of values for those actions using target_net.
        Q = sess.run(self.target_net, feed_dict={self.state_input: next_states})
        print('Q=\n', Q)
        doubleQ = Q[range(self.batch_size), actions]
        print('doubleQ=', doubleQ)
        # Get the "true" value, Y, of the (state, action) based on the next
        # (state, action).
        target_vals = self.sample_rewards() + \
                      self.sample_not_dones() * self.future_discount * doubleQ
        print('rewards=', self.sample_rewards())
        print('target_vals=', target_vals)
        print(' ')
        # Update weights in main_net.
        _ = sess.run(self.train_op,
                     feed_dict={
                         self.state_input: self.sample_states(),
                         self.taken_actions_input: self.sample_actions(),
                         self.target_vals_input: target_vals})
        self.update_target_net(sess)

    def update_target_net(self, sess):
        """
        Every tau updates, we will update the target net to be a copy of the
        main network. This is so that it will be able to reflect the learning
        that the main network has done.
        :param sess: tf.Session
        :return:
        """
        if self.update_count % self.update_target_net_rate != 0:
            return
        self.update_count = 0

        # Go through each layer in the NN and copy the weights from main_net to
        # target_net
        main_vars = tf.trainable_variables(scope="main_net")
        target_vars = tf.trainable_variables(scope="target_net")
        for main, target in zip(main_vars, target_vars):
            sess.run(target.assign(main.value()))

    def create_experiences(self, experiences_size):
        return BaseDoubleQnet.ExperienceBuffer(
            state=deque(maxlen=experiences_size),
            action=deque(maxlen=experiences_size),
            reward=deque(maxlen=experiences_size),
            next_state=deque(maxlen=experiences_size),
            done=deque(maxlen=experiences_size))

    # sample_x functions are a way to consistently draw a set of like values
    # from the experiences. To change the sampling set, you must update
    def new_sampling(self):
        # Creates a new set of random indices to sample experiences with.
        self.sample_idx = random.sample(range(len(self.experiences.state)),
                                        self.batch_size)

    def sample_states(self):
        return np.array([self.experiences.state[i] for i in self.sample_idx])

    def sample_actions(self):
        return np.array([self.experiences.action[i] for i in self.sample_idx])

    def sample_rewards(self):
        return np.array([self.experiences.reward[i] for i in self.sample_idx])

    def sample_next_states(self):
        return np.array(
            [self.experiences.next_state[i] for i in self.sample_idx])

    def sample_not_dones(self):
        return np.array([not self.experiences.done[i] for i in self.sample_idx])

    def clear_experiences(self):
        self.experiences.states.clear()
        self.experiences.actions.clear()
        self.experiences.dones.clear()
        self.experiences.next_states.clear()
        self.experiences.rewards.clear()

    @property
    def experiences_full(self):
        return len(self.experiences.state) == self.experiences.state.maxlen

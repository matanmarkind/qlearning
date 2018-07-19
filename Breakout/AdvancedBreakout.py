"""
This is an attempt to recreate the algorithm that was used by deepmind in the
first major paper they published about beating atari games.
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Uses some changes suggested in
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

And the gaps were filled based on the implementation in
https://github.com/boyuanf/DeepQLearning/blob/master/deep_q_learning.py
"""

from datetime import datetime
from resource import getrusage, RUSAGE_SELF

import tensorflow as tf
import numpy as np

import gym, os, argparse, sys, time

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)
from utils.ExperienceBuffer import WeightedExpBuf
from utils.BaseReplayQnet import BaseReplayQnet

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, help='train, show')
# TODO: consider having a 2 step anneal. Here we stop at 10% but that may
# make long terms planning hard for the network since the further into
# the future we go, the more likely its planning is to get messed up
# by a forced random action. Perhaps do 100% -> 10% over X steps, then
# hold random action at 10% for X steps, then anneal from 10% -> 1%
# over another X steps.
parser.add_argument('--e_i', type=float, default=1,
                    help="Initial chance of selecting a random action.")
parser.add_argument('--e_f', type=float, default=.1,
                    help="Final chance of selecting a random action.")
parser.add_argument(
    '--e_anneal', type=int, default=int(1e6),
    help='Number of updates to linearly anneal from e_i to e_f.')
parser.add_argument(
    '--ckpt_dir', type=str,
    help='Folder to save checkpoints to.')
parser.add_argument('--restore_ckpt', type=str,
                    help='path to restore a ckpt from')
parser.add_argument(
    '--exp_capacity', type=int, default=int(3e5),
    help='Number of past experiences to hold for replay. (300k ~ 10GB)')
parser.add_argument(
    '--begin_updates', type=int, default=int(1e5),
    help='Number of experiences before begin to training begins.')
parser.add_argument(
    '--batch_size', type=int, default=32,
    help='Batch size for each update to the network (multiple of 8)')
parser.add_argument(
    '--output_period', type=int, default=1000,
    help='Number of episodes between outputs (print, checkpoint)')
parser.add_argument(
    '--learning_rate', type=float, default=1e-3,
    help="learning rate for the network. passed to the optimizer.")
parser.add_argument(
    '--future_discount', type=float, default=0.99,
    help="Rate at which to discount future rewards.")


def preprocess_img(img):
    """
    Images are converted from RGB to grayscale and downsampled by a factor of
    2. Deepmind actually used a final 84x84 image by cropping since their GPU
    wanted a square input for convolutions. We do not preprocess, rather we
    store as uint8 for the sake of memory.
    :param img: Atari image (210, 160, 3)
    :return: Grayscale downsample version (105, 80)
    """
    return np.mean(img[::2, ::2], axis=2).astype(np.uint8)

def normalize(states):
    """

    :param states: numpy array of states
    :return: normalized img with values on [-1, 1)
    """
    return states.astype(np.float32) / 128. - 1

class AdvancedBreakoutQnet(BaseReplayQnet):
    """
    Class to perform basic Q learning
    """
    def __init__(self, input_shape, n_actions, batch_size, optimizer,
                 exp_buf_capacity, discount):
        BaseReplayQnet.__init__(
            self, input_shape, n_actions, batch_size, optimizer,
            WeightedExpBuf(exp_buf_capacity), discount)

    def make_nn_impl(self):
        """
        Make a NN to take in a batch of states (4 preprocessed images) with
        an output of size 3 (stay, left, right). No activation function is
        applied to the final output.
        :return:
        """
        initializer = tf.contrib.layers.xavier_initializer
        print('state_input', self.state_input)
        conv1 = tf.layers.conv2d(self.state_input, 16, (8, 8), (4, 4),
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer(),
                                 bias_initializer=initializer())
        print('conv1', conv1)
        conv2 = tf.layers.conv2d(conv1, 32, (4, 4), (2, 2),
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer(),
                                 bias_initializer=initializer())
        print('conv2', conv2)
        hidden = tf.layers.dense(tf.layers.flatten(conv2), 256,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer(),
                                 bias_initializer=initializer())
        print('hidden', hidden)
        return tf.layers.dense(hidden, self.n_actions,
                               kernel_initializer=initializer(),
                               bias_initializer=initializer())

    def loss_fn(self, expected, actual):
        """
        A function for calculating the loss of the neural network. Common
        examples include RootMeanSquare or HuberLoss.
        :param expected: a batch of target_vals
        :param actual: a batch of ouputs from the network
        :return: a batch of losses.
        """
        return tf.losses.mean_squared_error(
            labels=expected, predictions=actual,
            reduction=tf.losses.Reduction.NONE)

    def update(self, sess):
        """
        Perform a basic Q learning update by taking a batch of experiences from
        memory and replaying them.
        :param sess: tf.Session()
        """
        # Get a batch of past experiences.
        ids, states, actions, rewards, next_states, not_terminals = \
            self.exp_buf.sample(self.batch_size)
        states = normalize(states)
        next_states = normalize(next_states)

        # To predict the 'true' Q value, we use the network to predict the value
        # of the next_state, which is the value of the best action we can take
        # from the next state.
        next_actions = self.predict(sess, next_states)
        # Calculate the Q value for each of the next_states, and take the Q
        # value of the action we would take for each next_state.
        fullQ = sess.run(self.main_net,
                         feed_dict={self.state_input: next_states})
        nextQ = fullQ[range(self.batch_size), next_actions]

        # Discounted future value:
        # trueQ = r + discount * Q(next_state, next_action)
        # If this is a terminal term, trueQ = r
        target_vals = rewards + not_terminals * self.discount * nextQ

        # Calculate the value of being back in state and performing action. Then
        # compare that the the expected value just calculated. This is used to
        # compute the error for feedback. Then backpropogate the loss so that
        # the network can update.

        _ = sess.run(self.train_op,
                     feed_dict={
                         self.state_input: states,
                         self.action_input: actions,
                         self.target_vals_input: target_vals})

        # Now recalculate the loss of the network to update the weights of the experiences.
        next_actions = self.predict(sess, next_states)
        fullQ = sess.run(self.main_net,
                         feed_dict={self.state_input: next_states})
        nextQ = fullQ[range(self.batch_size), next_actions]
        target_vals = rewards + not_terminals * self.discount * nextQ
        loss = sess.run(self.loss,
                        feed_dict={
                         self.state_input: states,
                         self.action_input: actions,
                         self.target_vals_input: target_vals})
        self.exp_buf.update_weights(ids, loss)

def play_episode(args, sess, env, qnet, e):
    """
    Actually plays a single game and performs updates once we have enough
    experiences.
    :param args: parser.parse_args
    :param sess: tf.Session()
    :param env: gym.make()
    :param qnet: class which holds the NN to play and update.
    :param e: chance of a random action selection.
    :return: reward earned in the game, update value of e
    """
    done = False
    img = preprocess_img(env.reset())
    state = np.stack((img, img, img, img), axis=2)
    reward = 0  # total reward for this episode
    turn = 0

    while not done:
        action = qnet.predict(sess, normalize(np.array([state])))[0]
        if np.random.rand(1) < e:
            action = qnet.rand_action()

        img, r, done, _ = env.step(action + 1) # {1, 2, 3}
        img = np.reshape(preprocess_img(img), (105, 80, 1))
        next_state = np.concatenate((state[:, :, :3], img), axis=2)
        # TODO: r += info['ale.lives'] - old_lives??
        # Feels weird to clip the reward, but comes from Deepmind paper...
        qnet.add_experience(state, action, np.clip(r, -1, 1), next_state, done)

        if qnet.exp_buf_size() > args.begin_updates:
            # Once we have enough experiences in the buffer we can
            # start learning.
            if turn % (qnet.batch_size // 8) == 0:
                # We want to use each experience on average 8 times so
                # that's why for a batch size of 8 we would update every turn.
                qnet.update(sess)
            if e > args.e_f:
                # Reduce once for every update on 8 states. This makes e
                # not dependent on the batch_size.
                e -= (args.e_i - args.e_f) / args.e_anneal

        state = next_state
        reward += r
        turn += 1

    return reward, e, turn

def maybe_output(args, sess, saver, qnet, episode, e, rewards, turn):
    """
    Periodically we want to create some sort of output (printing, saving, etc...).
    This function does that.
    :param args: parser.parse_args
    :param sess: tf.Session()
    :param saver: tf.train.Saver()
    :param qnet: class which holds the NN used to play learn and which holds the
        experiences.
    :param episode: Episode number
    :param e: chance of random action
    :param rewards: list of rewards for each episode played.
    :return:
    """

    if (episode +1) % args.output_period != 0:
        return

    # Print info about the state of the network
    turn_str =' turn=' + str(turn)
    e_str = ' e={:0.2f}'.format(e)
    mem_usg_str = \
        ' mem_usage={:0.2f}GB'.format(getrusage(RUSAGE_SELF).ru_maxrss / 2**20)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S "), mem_usg_str,
          ' episode=', episode+1,
          ' reward_last_' + str(args.output_period) + '_games=',
          int(sum(rewards[-args.output_period:])), e_str, turn_str,
          sep='')

    # save the model
    model_name = 'model-AdvancedBreakout-' + str(episode+1) + '.ckpt'
    saver.save(sess, os.path.join(args.ckpt_dir, model_name))

def get_qnet(args, scope=''):
    """
    Wrapper for getting the Gridworld network so don't have to copy and paste
    the same params each time.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        return AdvancedBreakoutQnet(
            input_shape = (105, 80, 4), n_actions=3, batch_size=args.batch_size,
            optimizer=tf.train.RMSPropOptimizer(.00025, decay=.95, epsilon=.01),
            exp_buf_capacity=args.exp_capacity, discount=args.future_discount)

def train(args):
    """
    This function trains a Neural Network on how to play brickbreaker. Is
    meant to be identical to how Deepminds paper "Playing Atari with Deep
    Reinforcement Learning" works.
    :param args: parser.parse_args
    :return:
    """
    env = gym.make('BreakoutDeterministic-v4')
    tf.reset_default_graph()
    qnet = get_qnet(args)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Don't want to change the graph once we begin playing the game.
    tf.get_default_graph().finalize()
    with tf.Session(config=tf.ConfigProto(operation_timeout_in_ms=10000)) as sess:
        sess.run(init)
        e = args.e_i
        episode = 0
        rewards = []
        turn = 0

        while True:
            r, e, t = play_episode(args, sess, env, qnet, e)
            turn += t
            rewards.append(r)

            episode += 1
            maybe_output(args, sess, saver, qnet, episode, e, rewards, turn)

def show_game(args):
    env = gym.make('BreakoutDeterministic-v4')
    tf.reset_default_graph()
    qnet = get_qnet(args)

    saver = tf.train.Saver()
    tf.get_default_graph().finalize()
    with tf.Session(config=tf.ConfigProto(operation_timeout_in_ms=10000)) as sess:
        saver.restore(sess, args.restore_ckpt)
        done = False
        img = preprocess_img(env.reset())
        _ = env.render()
        state = np.stack((img, img, img, img), axis=2)
        reward, turns = 0, 0

        while not done:
            time.sleep(.25)
            action = qnet.predict(sess, normalize(np.array([state])))[0]

            img, r, done, _ = env.step(action + 1) # {1, 2, 3}
            _ = env.render()
            img = np.reshape(preprocess_img(img), (105, 80, 1))
            state = np.concatenate((state[:, :, :3], img), axis=2)
            reward += r
            turns += 1

    print('turns =', turns, ' reward =', reward)

def main():
    args = parser.parse_args(sys.argv[1:])
    if args.mode == 'show':
        assert args.restore_ckpt != '', 'Must provide a checkpoint to show.'
        args.exp_capacity = 0
        show_game(args)
    elif args.mode == 'train':
        assert args.ckpt_dir != '', \
            'Must provide a directory to save checkpoints to.'
        train(args)
    else:
        assert False, "Must provide a mode to run in: train, show."

if __name__ == '__main__':
    main()


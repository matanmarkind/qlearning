"""
This is an attempt to create a more advanced Q net to beat Breakout. Still
based largely on the deepmind paper, but trying to work in some other
features
    - Huber loss
    - Weighted experience replay
    - Double DQN
    - Deuling networks
    - Updated reward function
    - Adamoptimizer
"""

from BaseReplayQnet import  BaseReplayQnet
from ExperienceBuffer import WeightedExpBuf
from datetime import datetime
from resource import getrusage, RUSAGE_SELF

import tensorflow as tf
import numpy as np

import gym, os, argparse, sys, time

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='show',
                    help='train, show')
parser.add_argument('--e_i', type=float, default=1,
                    help="Initial chance of selecting a random action.")
parser.add_argument('--e_f', type=float, default=.1,
                    help="Final chance of selecting a random action.")
parser.add_argument(
    '--e_anneal', type=int, default=int(1e6),
    help='Number of updatews to linearly anneal from eps_i to eps_f.')
parser.add_argument(
    '--ckpt_dir', type=str,
    default=os.path.join('..', 'models', 'Breakout'),
    help='Folder to save checkpoints to.')
parser.add_argument('--ckpt_path', type=str,
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
parser.add_argument('--future_reward_discount', type=float, default=.99,
                    help="Exponential decay rate of future rewards")

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
    :return:
    """
    return states.astype(np.float32) / 255.

class MatanBreakoutQnet(BaseReplayQnet):
    """
    Class to perform basic Q learning
    """
    def __init__(self, input_shape, n_actions, batch_size,
                 optimizer, exp_buf_capacity, update_target_net_rate, discount):
        BaseReplayQnet.__init__(
            self, input_shape, n_actions, batch_size, optimizer, exp_buf_capacity,
            WeightedExpBuf, discount)

        self.replay_count = 0  # Number of experiences replayed
        self.update_target_net_rate = update_target_net_rate

        # This NN evaluates the value of the action selected by the
        # main_net. This is the secondary net that will be updated by copying
        # from the main_net every tau updates (tau = update_target_net)
        self.target_net = self.make_nn('target_net')

    def make_nn_impl(self):
        """
        Make a NN to take in a batch of states (4 preprocessed images) with
        an output of size 3 (stay, left, right). No activation function is
        applied to the final output.
        :return:
        """
        # Use a convolutional network to analyze the state which is a set of
        # normalized images.
        initializer = tf.contrib.layers.xavier_initializer
        conv1 = tf.layers.conv2d(self.state_input, 32, (8, 8), (4, 4),
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer(),
                                 bias_initializer=initializer())
        conv2 = tf.layers.conv2d(conv1, 64, (4, 4), (2, 2),
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer(),
                                 bias_initializer=initializer())
        conv3 = tf.layers.conv2d(conv2, 64, (3, 3), (1, 1),
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer(),
                                 bias_initializer=initializer())
        hidden = tf.layers.dense(tf.layers.flatten(conv3), 512,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer(),
                                 bias_initializer=initializer())

        # Dueling Qnet. This allows the netowrk to learn the value of being in
        # a given state and separately to learn which action taken in that state
        # will be the most advantageous.
        v, a = tf.split(hidden, 2, 1)
        value = tf.layers.dense(inputs=v, units=1,
                                kernel_initializer=initializer(),
                                bias_initializer=initializer())
        advantage = tf.layers.dense(inputs=a, units=self.n_actions,
                                    kernel_initializer=initializer(),
                                    bias_initializer=initializer())

        # advantage is used to determine the relative advantage of any move.
        # value is the base value of just being in this state.
        adv_offset = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return value + advantage - adv_offset

    def loss(self, expected, actual):
        """
        A function for calculating the loss of the neural network. Common
        examples include RootMeanSquare or HuberLoss.
        :param expected: a batch of target_vals
        :param actual: a batch of ouputs from the network
        :return: a batch of losses.
        """
        return tf.reduce_mean(tf.square(expected - actual))

    def update(self, sess):
        """
        Perform a basic Q learning update by taking a batch of experiences from
        memory and replaying them.
        :param sess: tf.Session()
        """
        self.replay_count += self.batch_size

        # Get a batch of past experiences.
        exp_ids, states, actions, rewards, next_states, not_terminals = \
            self.exp_buf.sample(self.batch_size)
        states = normalize(states)
        next_states = normalize(next_states)

        # To predict the 'true' Q value, we use the network to predict the value
        # of the next_state, which is the value of the best action we can take
        # from the next state.
        next_actions = self.predict(sess, next_states)
        # Calculate the Q value for each of the next_states, and take the Q
        # value of the action we would take for each next_state.
        fullQ = sess.run(self.target_net,
                         feed_dict={self.state_input: next_states})
        nextQ = fullQ[:, next_actions]

        # Discounted future value:
        # If this is a terminal term: trueQ = r
        # else: trueQ = r + discount * Q(next_state, next_action)
        target_vals = rewards + not_terminals * self.discount * nextQ

        # Calculate the value of being back in state and performing action. Then
        # compare that the the expected value just calculated. This is used to
        # compute the error for feedback.
        loss, _ = sess.run([self.loss, self.train_op],
                           feed_dict={
                               self.state_input: states,
                               self.taken_actions_input: actions,
                               self.target_vals_input: target_vals})

        # TODO: verify loss is in the same order as exp_ids
        self.exp_buf.update_weights(exp_ids, loss)

        self.update_target_net()

    def update_target_net(self):
        if self.replay_count < self.update_target_net_rate:
            return
        self.replay_count = 0
        # TODO: Copy main_net to target_net


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

        img, r, done, _ = env.step(action + 1) # 0 & 1 don't do anything
        img = np.reshape(preprocess_img(img), (105, 80, 1))
        next_state = np.concatenate((state[:, :, :3], img), axis=2)
        # TODO: r += info['ale.lives'] - old_lives??
        # Feels weird to clip the reward, but comes from Deepmind paper...
        qnet.add_experience(state, action, np.clip(r, -1, 1), next_state, done)

        if turn % (qnet.batch_size // 8) == 0 and qnet.exp_buf_size() > args.begin_updates:
            # Once we have enough experiences in the buffer we can
            # start learning. We want to use each experience on average 8 times
            # so that is why for a batch size of 8 we would update every turn.
            qnet.update(sess)
            if e > args.e_f:
                e -= (args.e_i - args.e_f) / args.e_anneal

        state = next_state
        reward += r
        turn += 1

    return reward, e

def maybe_output(args, sess, saver, qnet, episode, e, rewards):
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
    exp_buf_size = qnet.exp_buf_size()
    exp_buf_capacity = qnet.exp_buf_capacity()
    turn_str = \
        ' turn=' + str(exp_buf_size) if exp_buf_size < exp_buf_capacity else ''
    e_str = '' if e > args.e_f else str(e)
    mem_usg_str = \
        ' mem_usage={:0.2f}GB'.format(getrusage(RUSAGE_SELF).ru_maxrss / 2**20)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S "), mem_usg_str,
          ' episode=', episode+1,
          ' reward_last_' + str(args.output_period) + '_games=',
          int(sum(rewards[-args.output_period:])), turn_str, e_str,
          sep='')

    # save the model
    model_name = 'model-deepmind-' + str(episode+1) + '.ckpt'
    saver.save(sess, os.path.join(args.ckpt_dir, model_name))

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
    qnet = DeepmindBreakoutQnet(
        input_shape = (105, 80, 4), n_actions=3, batch_size=args.batch_size,
        optimizer=tf.train.RMSPropOptimizer(.00025, decay=.95, epsilon=.01),
        exp_buf_capacity=args.exp_capacity)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Don't want to change the graph once we begin playing the game.
    tf.get_default_graph().finalize()
    with tf.Session(config=tf.ConfigProto(operation_timeout_in_ms=10000)) as sess:
        sess.run(init)
        e = args.e_i
        episode = 0
        rewards = []

        while True:
            r, e = play_episode(args, sess, env, qnet, e)
            rewards.append(r)

            episode += 1
            maybe_output(args, sess, saver, qnet, episode, e, rewards)

def show_game(args):
    env = gym.make('BreakoutDeterministic-v4')
    tf.reset_default_graph()
    qnet = DeepmindBreakoutQnet(
        input_shape = (105, 80, 4), n_actions=3, batch_size=args.batch_size,
        optimizer=tf.train.RMSPropOptimizer(.00025, decay=.95, epsilon=.01),
        exp_buf_capacity=args.exp_capacity)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    tf.get_default_graph().finalize()
    with tf.Session(config=tf.ConfigProto(operation_timeout_in_ms=10000)) as sess:
        saver.restore(sess, args.ckpt_path)
        done = False
        img = preprocess_img(env.reset())
        _ = env.render()
        state = np.stack((img, img, img, img), axis=2)

        while not done:
            time.sleep(.25)
            action = qnet.predict(sess, normalize(np.array([state])))[0]

            img, r, done, _ = env.step(action + 1) # 0 and 1 don't do anything
            _ = env.render()
            img = np.reshape(preprocess_img(img), (105, 80, 1))
            state = np.concatenate((state[:, :, :3], img), axis=2)

def main():
    args = parser.parse_args(sys.argv[1:])
    if args.mode == 'show':
        assert args.ckpt_path != '', 'Must provide a checkpoint to show.'
        show_game(args)
    elif args.mode == 'train':
        train(args)

if __name__ == '__main__':
    main()
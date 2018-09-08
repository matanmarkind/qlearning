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
from matplotlib import pyplot as plt

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
parser.add_argument('--e_i', type=float, default=1,
                    help="Initial chance of selecting a random action.")
parser.add_argument('--e_f', type=float, default=.05,
                    help="Final chance of selecting a random action.")
parser.add_argument(
    '--e_anneal', type=int, default=int(10e6),
    help='Number of transition replays over which to linearly anneal from e_i '
         'to e_f.')
parser.add_argument(
    '--ckpt_dir', type=str,
    help='Folder to save checkpoints to.')
parser.add_argument('--restore_ckpt', type=str,
                    help='path to restore a ckpt from')
parser.add_argument(
    '--exp_capacity', type=int, default=int(6e5),
    help='Number of past experiences to hold for replay. (600k ~ 12.5GB)')
parser.add_argument(
    '--begin_updates', type=int, default=int(2e5),
    help='Number of experiences before begin to training begins.')
parser.add_argument(
    '--batch_size', type=int, default=32,
    help='Batch size for each update to the network (multiple of 8)')
parser.add_argument(
    '--output_period', type=int, default=int(2e6),
    help='Number of transition updates between outputs (print, checkpoint)')
parser.add_argument(
    '--learning_rate', type=float, default=1e-4,
    help="learning rate for the network. passed to the optimizer.")
parser.add_argument(
    '--future_discount', type=float, default=0.99,
    help="Rate at which to discount future rewards.")
parser.add_argument('--train_record_fname', type=str,
        default="training-record-RainbowBreakout.txt",
        help="Absolute path to file to save progress to (same as what is"
        " printed to cmd line.")
parser.add_argument('--train_steps', type=int, default=int(100e6),
                    help="Number of transition replays to experience "
                         "(will update train_steps // batch_size times)")
parser.add_argument(
    '--show_random', type=bool, default=False,
    help="Use random actions when mode=show at a rate of e_f")
parser.add_argument(
    '--random_starts', type=int, default=30,
    help='randomly perform stand still at beginning of episode.')
parser.add_argument('--alpha', type=float, default=.6,
                    help="Factor for how much weight prioritization")
parser.add_argument('--beta_i', type=float, default=.4,
                    help="initial weighting for bias correction")
parser.add_argument('--beta_f', type=float, default=1,
                    help="final weighting for bias correction")
# Not sure what value makes sense for the weight offset. Does having an
# absolute offset even make sense at all? The absolute value of the loss
# is fairly meaningless, given I am performing a classification with no
# normalization layer.
parser.add_argument('--priority_weight_offset', type=float, default=.01,
                    help="small value so no transition has 0 weight.")


def preprocess_img(img):
    """
    Images are converted from RGB to grayscale and downsampled by a factor of
    2. Deepmind actually used a final 84x84 image by cropping since their GPU
    wanted a square input for convolutions. We do not preprocess, rather we
    store as uint8 for the sake of memory.
    :param img: Atari image (210, 160, 3)
    :return: Grayscale downsample version (85, 80)
    """
    return np.mean(img[::2, ::2], axis=2).astype(np.uint8)[15:100, :]

def normalize(states):
    """

    :param states: numpy array of states
    :return: normalized img with values on [-1, 1)
    """
    return states.astype(np.float32) / 128. - 1

class RainbowBreakoutQnet(BaseReplayQnet):
    """
    Class to perform basic Q learning
    """
    def __init__(self, input_shape, n_actions, batch_size, optimizer,
                 exp_buf_capacity, discount, alpha, beta_i, beta_f, beta_anneal,
                 weight_offset, is_training):
        self.is_training = is_training
        exp_buf = WeightedExpBuf(exp_buf_capacity, alpha, beta_i, beta_f,
                                 beta_anneal, weight_offset)
        BaseReplayQnet.__init__(
            self, input_shape, n_actions, batch_size, optimizer, exp_buf,
            discount)

    def make_nn_impl(self):
        """
        Make a NN to take in a batch of states (3 preprocessed images) with
        an output of size 3 (stay, left, right). No activation function is
        applied to the final output.
        :return:
        """
        if (self.is_training):
            print('using dropout')

        print('state_input', self.state_input)
        conv1 = tf.layers.conv2d(self.state_input, 32, (8, 8), (4, 4),
                                 activation=tf.nn.relu)
        print('conv1', conv1)
        conv2 = tf.layers.conv2d(conv1, 64, (4, 4), (2, 2),
                                 activation=tf.nn.relu)
        print('conv2', conv2)
        hidden1 = tf.layers.dense(tf.layers.flatten(conv2), 256,
                                  activation=tf.nn.relu)
        print('hidden1', hidden1)
        dropout1 = tf.layers.dropout(hidden1, training=self.is_training)
        print('dropout1', dropout1)
        return tf.layers.dense(dropout1, self.n_actions)

    def loss_fn(self, expected, actual):
        """
        A function for calculating the loss of the neural network.
        IS_weights_input must be created here since loss_fn is an abstractmethod
        and must be available to the parent, so dependencies cannot be pushed
        off until after parent construction. (Seems like a bad design in python,
        a virtual function should be allowed to depend on things specific to
        the child/implementer class).

        :param expected: a batch of target_vals
        :param actual: a batch of ouputs from the network
        :return: a batch of losses.
        """
        # When using priority replay the network sees surprising events
        # more often. These events, by their nature, tend to have larger
        # than median errors. This combination of seeing larger loss
        # events at a larger frequency, along with the variable priority of
        # the transitions, adds a bias to the network. Importance sampling
        # is used to correct for this, downweighting the loss for the events
        # that will be selected more often.
        self.IS_weights_input = tf.placeholder(shape=None, dtype=tf.float32)

        return tf.losses.huber_loss(
            labels=expected, predictions=actual,
            weights=self.IS_weights_input,
            reduction=tf.losses.Reduction.NONE)

    def update(self, sess):
        """
        Perform a basic Q learning update by taking a batch of experiences from
        memory and replaying them.
        :param sess: tf.Session()
        """
        # Get a batch of past experiences.
        ids, states, actions, rewards, next_states, not_terminals, IS_weights =\
            self.exp_buf.sample(self.batch_size)
        states = normalize(states)
        next_states = normalize(next_states)

        # Calculate the predicted value from the network based on a previously
        # experienced state, assuming we perform the same action.
        fullQ = sess.run(self.main_net,
                         feed_dict={self.state_input: next_states})
        pred_vals = fullQ[range(self.batch_size), actions]

        # To predict the 'true' Q value, we use the network to predict the value
        # of the next_state, which is the value of the best action we can take
        # when in that state. Then take the value that the network predicts
        # we can get from the next_state.
        next_actions = self.predict(sess, next_states)
        fullQ = sess.run(self.main_net,
                         feed_dict={self.state_input: next_states})
        nextQ = fullQ[range(self.batch_size), next_actions]

        # Discounted future value:
        # trueQ = r + discount * Q(next_state, next_action)
        # If this is a terminal term, trueQ = r
        target_vals = rewards + not_terminals * self.discount * nextQ

        # Update the weighting for each experience based on its TD error.
        # TODO: why do we take the error before training, don't these weights
        # become stale as soon as train_op is run?
        self.exp_buf.update_weights(ids, abs(target_vals - pred_vals))

        # Calculate the value of being back in state and performing action. Then
        # compare that the the expected value just calculated. This is used to
        # compute the error for feedback. Then backpropogate the loss so that
        # the network can update.
        _ = sess.run(self.train_op,
                     feed_dict={
                         self.state_input: states,
                         self.action_input: actions,
                         self.target_vals_input: target_vals,
                         self.IS_weights_input: IS_weights})

def get_qnet(args, scope=''):
    """
    Wrapper for getting the Breakout network so don't have to copy and paste
    the same params each time.
    """
    assert args.batch_size % 8 == 0, "batch_size must be a multiple of 8"
    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        return RainbowBreakoutQnet(
            input_shape = (85, 80, 3), n_actions=3, batch_size=args.batch_size,
            optimizer=optimizer, exp_buf_capacity=args.exp_capacity,
            discount=args.future_discount, alpha=args.alpha, beta_i=args.beta_i,
            beta_f=args.beta_f, beta_anneal=args.train_steps // args.batch_size,
            weight_offset=args.priority_weight_offset,
            is_training= (args.mode == 'train'))

def play_episode(args, sess, env, qnet, e):
    """
    Actually plays a single game and performs updates once we have enough
    experiences.
    :param args: parser.parse_args
    :param sess: tf.Session()
    :param env: gym.make()
    :param qnet: class which holds the NN to play and update.
    :param e: chance of a random action selection.
    :return: reward earned in the game, update value of e, transitions updated
        against.
    """
    done = False
    _ = env.reset()
    reward = 0  # total reward for this episode
    turn = 0
    transitions = 0  # updates * batch_size
    lives = 5  # Always start with 5 lives
    terminal = True  # Anytime we lose a life, and beginning of episode.

    while not done:
        if terminal:
            terminal = False
            # To make sure that the agent doesn't just learn to set up well for
            # the way the game starts, begin the game by not doing anything and
            # letting the ball move.
            for _ in range(np.random.randint(1, args.random_starts)):
                img, _, done, info = env.step(1)  # starts game, but stays still
            img = preprocess_img(img)
            state = np.stack((img, img, img), axis=2)

        action = qnet.predict(sess, normalize(np.array([state])))[0]
        if np.random.rand(1) < e:
            action = qnet.rand_action()

        # Perform an action, prep the data, and store as an experience
        img, r, done, info = env.step(action + 1)  # {1, 2, 3}
        img = np.reshape(preprocess_img(img), (85, 80, 1))
        next_state = np.concatenate((state[:, :, 1:], img), axis=2)
        if info['ale.lives'] < lives:
            terminal = True
            lives = info['ale.lives']
        qnet.add_experience(state, action, r, next_state, terminal)

        if qnet.exp_buf_size() > args.begin_updates:
            # Once we have enough experiences in the buffer we can
            # start learning.
            if turn % (qnet.batch_size // 8) == 0:
                # We want to use each experience on average 8 times so
                # that's why for a batch size of 8 we would update every turn.
                qnet.update(sess)
                transitions += qnet.batch_size
                if e > args.e_f:
                    # Reduce once for every update on 8 states. This makes e
                    # not dependent on the batch_size.
                    e -= (qnet.batch_size*(args.e_i - args.e_f)) / args.e_anneal

        state = next_state
        reward += r
        turn += 1

    return reward, e, transitions

def write_output(args, sess, saver, last_output_ep, e, rewards,
                 transitions, qnet):
    """
    Periodically we want to create some sort of output (printing, saving, etc...).
    This function does that.
    :param args: parser.parse_args
    :param sess: tf.Session()
    :param saver: tf.train.Saver()
    :param last_output_ep: number of episodes played at last output
    :param e: chance of random action
    :param rewards: list of rewards for each episode played.
    :param transitions: Number of transitions replayed.
    :param qnet: NN being trained
    :return:
    """
    num_eps = len(rewards) - last_output_ep

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mem_usg_str = \
        'mem_usage={:0.2f}GB'.format(getrusage(RUSAGE_SELF).ru_maxrss / 2**20)
    episode_str = 'episode=' + str(len(rewards))
    reward_str = 'avg_reward_last_' + str(num_eps) + '_games=' + \
                 str(sum(rewards[-num_eps:]) // num_eps)
    e_str = 'e={:0.2f}'.format(e)
    transitions_str ='training_step=' + str(transitions)
    beta_str = 'beta={:0.3f}'.format(qnet.exp_buf.beta)

    output_str = '  '.join((time_str, mem_usg_str, episode_str, reward_str,
                            e_str, transitions_str, beta_str))
    print(output_str)
    with open(os.path.join(args.ckpt_dir, args.train_record_fname), 'a') as f:
        f.write(output_str + '\n')

    # save the model
    model_name = 'model-RainbowBreakout-' + str(transitions) + '.ckpt'
    saver.save(sess, os.path.join(args.ckpt_dir, model_name))

def train(args):
    """
    This function trains a Neural Network on how to play brickbreaker. Is
    meant to be identical to how Deepminds paper "Playing Atari with Deep
    Reinforcement Learning" works. I use 3 images to make the state instead
    of 4 since when using 4 I only had enough for 400k states in the buffer,
    but now I can fit 600k and it still does well.
    :param args: parser.parse_args
    :return:
    """
    with open(os.path.join(args.ckpt_dir, args.train_record_fname), 'a') as f:
        f.write("RainbowBreakout -- begin training --\n")

    tf.reset_default_graph()
    env = gym.make('BreakoutDeterministic-v4')
    qnet = get_qnet(args)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Don't want to change the graph once we begin playing the game.
    tf.get_default_graph().finalize()
    with tf.Session() as sess:
        sess.run(init)
        e = args.e_i
        last_output_ep = 0
        rewards = []
        transitions = 0  # number of transitions updated against
        next_output = args.output_period

        while transitions < args.train_steps:
            r, e, t = play_episode(args, sess, env, qnet, e)
            if transitions == 0 and t > 0:
                # Output status from before training starts.
                write_output(args, sess, saver, last_output_ep, e, rewards,
                             transitions, qnet)
                last_output_ep = len(rewards)

            transitions += t
            rewards.append(r)

            if transitions > next_output:
                # Regular output during training.
                write_output(args, sess, saver, last_output_ep, e, rewards,
                             transitions, qnet)
                next_output += args.output_period
                last_output_ep = len(rewards)

    with open(os.path.join(args.ckpt_dir, args.train_record_fname), 'a') as f:
        f.write('\n\n')

def show_game(args):
    plt.ion()
    plt.show()
    figure = plt.figure()
    env = gym.make('BreakoutDeterministic-v4')
    tf.reset_default_graph()
    qnet = get_qnet(args)

    saver = tf.train.Saver()
    tf.get_default_graph().finalize()
    with tf.Session(config=tf.ConfigProto(operation_timeout_in_ms=10000)) as sess:
        saver.restore(sess, args.restore_ckpt)
        done = False
        img = env.reset()
        _ = env.render()
        img = preprocess_img(img)
        state = np.stack((img, img, img), axis=2)
        reward, turns = 0, 0

        while not done:
            t1 = time.time()
            action = qnet.predict(sess, normalize(np.array([state])))[0]
            if args.show_random and np.random.rand(1) < args.e_f:
                action = 0  # Doesn't seem to like restarting

            img, r, done, _ = env.step(action + 1) # {1, 2, 3}
            _ = env.render()
            img = np.reshape(preprocess_img(img), (85, 80, 1))
            state = np.concatenate((state[:, :, 1:], img), axis=2)
            reward += r
            turns += 1

            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.imshow(state[:, :, i])
            plt.draw()
            plt.pause(max(0, .025 - (time.time() - t1)))

    print('turns =', turns, ' reward =', reward)
    plt.close()
    plt.ioff()

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

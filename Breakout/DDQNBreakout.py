"""
This is an attempt to recreate the deepmind DDQN algorithm to beat atari games:
https://arxiv.org/pdf/1509.06461.pdf

article and code that guided this attempt:
https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
"""

from datetime import datetime
from resource import getrusage, RUSAGE_SELF

import tensorflow as tf
import numpy as np
import gym, os, argparse, sys, time

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)
from utils.ExperienceBuffer import ExpBuf
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
    '--exp_capacity', type=int, default=int(4e5),
    help='Number of past experiences to hold for replay. (400k ~ 11GB)')
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
    '--learning_rate', type=float, default=1e-5,
    help="learning rate for the network. passed to the optimizer.")
parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                    help='Constant used by AdamOptimizer')
parser.add_argument(
    '--future_discount', type=float, default=0.99,
    help="Rate at which to discount future rewards.")
parser.add_argument('--train_record_fname', type=str,
        default="training-record-DDQNBreakout.txt",
        help="Absolute path to file to save progress to (same as what is"
        " printed to cmd line.")
parser.add_argument('--train_steps', type=int, default=int(100e6),
                    help="Number of transition replays to experience "
                         "(will update train_steps // batch_size times)")
parser.add_argument(
    '--update_target_net_period', type=int, default=int(8e4),
    help='Number of transition updates between copying main_net to target_net')
parser.add_argument(
    '--show_random', type=bool, default=False,
    help="Use random actions when mode=show at a rate of e_f")
parser.add_argument(
    '--random_starts', type=int, default=30,
    help='randomly perform stand still at beginning of episode.')


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

class DDQNBreakoutQnet(BaseReplayQnet):
    """
    Class to perform basic Q learning
    """
    def __init__(self, input_shape, n_actions, batch_size, optimizer,
                 exp_buf_capacity, discount, update_target_net_period):
        """

        :param input_shape:
        :param n_actions:
        :param batch_size:
        :param optimizer:
        :param exp_buf_capacity:
        :param discount:
        :param update_target_net: Number of updates between copying main_net to target_net.
        """
        BaseReplayQnet.__init__(
            self, input_shape, n_actions, batch_size, optimizer,
            ExpBuf(exp_buf_capacity), discount)
        self.target_net = self.make_nn('target_net')
        self.transition_updates = 0
        self.update_target_net_period = update_target_net_period

        main_vars = tf.trainable_variables("main_net")
        target_vars = tf.trainable_variables("target_net")
        self.update_target_net_ops = [t_var.assign(m_var.value())
                                      for m_var, t_var
                                      in zip(main_vars, target_vars)]

    def make_nn_impl(self):
        """
        Make a NN to take in a batch of states (3 preprocessed images) with
        an output of size 3 (stay, left, right). No activation function is
        applied to the final output.
        :return:
        """
        # TODO: think about putting the normalization here so don't need to
        # worry about it everywhere we use the NN.
        print('state_input', self.state_input)

        init = tf.variance_scaling_initializer(scale=2)
        conv1 = tf.layers.conv2d(
            self.state_input, filters=32, kernel_size=(8, 8), strides=4,
            activation=tf.nn.relu, kernel_initializer=init, use_bias=False,
            name="conv1")
        print('conv1', conv1)
        conv2 = tf.layers.conv2d(
            conv1, filters=64, kernel_size=(4, 4), strides=2, use_bias=False,
            activation=tf.nn.relu, kernel_initializer=init, name="conv2")
        print('conv2', conv2)
        conv3 = tf.layers.conv2d(
            conv2, filters=64, kernel_size=(3, 3), strides=1, use_bias=False,
            activation=tf.nn.relu, kernel_initializer=init, name="conv3")
        print('conv3', conv3)
        conv4 = tf.layers.conv2d(
            conv3, filters=1024, kernel_size=(7, 6), strides=1, use_bias=False,
            activation=tf.nn.relu, kernel_initializer=init, name="conv4")
        print('conv4', conv4)

        # Deuling networks - split now into value network, which should learn
        # the value of being in a given state, and advantage network, which
        # should learn the relative advantage of each possible action.
        vstream, astream = tf.split(conv4, 2, 3)
        vstream = tf.layers.flatten(vstream)
        print('vstream', vstream)
        astream = tf.layers.flatten(astream)
        print('astream', astream)
        value = tf.layers.dense(
            vstream, units=1, kernel_initializer=init, name="value")
        print('value', value)
        advantage = tf.layers.dense(
            astream, units=self.n_actions, kernel_initializer=init,
            name="advantage")
        print('advantage', advantage)
        print()  # Add empty line after finishing a network.

        # Subtract the average advantage since advantage should only be used to
        # differentiate between actions, not change the net value of the we
        # expect to get from this state.
        avg_advantage = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return value + advantage - avg_advantage

    def loss_fn(self, expected, actual):
        """
        A function for calculating the loss of the neural network. Common
        examples include RootMeanSquare or HuberLoss.
        :param expected: a batch of target_vals
        :param actual: a batch of ouputs from the network
        :return: a batch of losses.
        """
        return tf.losses.huber_loss(labels=expected, predictions=actual,
                                    reduction=tf.losses.Reduction.NONE)

    def update(self, sess):
        """
        Perform a basic Q learning update by taking a batch of experiences from
        memory and replaying them.
        :param sess: tf.Session()
        """

        # Every T updates, copy the main_net, which is the one being updated,
        # to target_net so that the Q value predictions are up to date. Done
        # before frame_update+= so that on the first update main_net is copied
        # to target_net.
        self.update_target_net(sess)
        self.transition_updates += self.batch_size

        # Get a batch of past experiences.
        states, actions, rewards, next_states, not_terminals = \
            self.exp_buf.sample(self.batch_size)
        states = normalize(states)
        next_states = normalize(next_states)

        # Double DQN - To predict the 'true' Q value, we use main_net to predict
        # the action we should take in the next state, and use target_net to
        # predict the value we expect to get from taking that action.
        next_actions = self.predict(sess, next_states)
        fullQ = sess.run(self.target_net,
                         feed_dict={self.state_input: next_states})
        nextQ = fullQ[range(self.batch_size), next_actions]

        # Discounted future value:
        # trueQ = r + discount * Q_target(next_state, next_action)
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

    def update_target_net(self, sess):
        if self.transition_updates % self.update_target_net_period == 0:
            for copy_op in self.update_target_net_ops:
                sess.run(copy_op)


def get_qnet(args, scope=''):
    """
    Wrapper for getting the Breakout network so don't have to copy and paste
    the same params each time.
    """
    assert args.batch_size % 8 == 0, "batch_size must be a multiple of 8"

    optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate,
                                     epsilon=args.adam_epsilon)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        return DDQNBreakoutQnet(
            input_shape = (85, 80, 4), n_actions=4, batch_size=args.batch_size,
            optimizer=optimizer, exp_buf_capacity=args.exp_capacity,
            update_target_net_period=args.update_target_net_period,
            discount=args.future_discount)

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
    terminal = True  # Anytime we lose a life, and beginning of episode.

    while not done:
        if terminal:
            terminal = False
            # To make sure that the agent doesn't just learn to set up well for
            # the way the game starts, begin the game by not doing anything and
            # letting the ball move.
            for _ in range(np.random.randint(1, args.random_starts)):
                # Perform random actions at the beginning so the network doesn't
                # just learn a sequence of steps to always take.
                img, _, done, info = env.step(env.action_space.sample())
            img = preprocess_img(img)
            state = np.stack((img, img, img, img), axis=2)
            lives = info['ale.lives']
        if done:
            # If lost our last life during random_start, nothing left to play
            break

        # Perform an action
        action = qnet.predict(sess, normalize(np.array([state])))[0]
        if np.random.rand(1) < e:
            action = qnet.rand_action()
        img, r, done, info = env.step(action)

        # Store as an experience
        img = np.reshape(preprocess_img(img), (85, 80, 1))
        next_state = np.concatenate((state[:, :, 1:], img), axis=2)
        if info['ale.lives'] < lives:
            terminal = True
        qnet.add_experience(state, action, r, next_state, terminal)

        # Updates
        if qnet.exp_buf_size() > args.begin_updates and\
                turn % (qnet.batch_size // 8) == 0:
            # Once we have enough experiences in the buffer we can
            # start learning. We want to use each experience on average 8 times
            # so that's why for a batch size of 8 we would update every turn.
            qnet.update(sess)
            transitions += qnet.batch_size
            if e > args.e_f:
                # Reduce once for every update on 8 states. This makes e
                # not dependent on the batch_size.
                e -= (qnet.batch_size*(args.e_i - args.e_f)) / args.e_anneal

        # Prep for the next turn
        state = next_state
        reward += r
        turn += 1

    return reward, e, transitions

def write_output(args, sess, saver, last_output_ep, e, rewards,
                 transitions):
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

    output_str = '  '.join((time_str, mem_usg_str, episode_str, reward_str,
                            e_str, transitions_str))
    print(output_str)
    with open(os.path.join(args.ckpt_dir, args.train_record_fname), 'a') as f:
        f.write(output_str + '\n')

    # save the model
    model_name = 'model-DDQNBreakout-' + str(transitions) + '.ckpt'
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
        f.write("DDQNBreakout -- begin training -- " + str(args) + "\n")
    # TODO: Figure out a way to only hold each img once and then reconstruct
    # the states from pointers to them. Would probs point to the last and grab
    # most_recent[-3:] and repeat an initial state 4x in the buffer like we
    # do to create the initial state now.
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
                             transitions)
                last_output_ep = len(rewards)

            transitions += t
            rewards.append(r)

            if transitions > next_output:
                # Regular output during training.
                write_output(args, sess, saver, last_output_ep, e, rewards,
                             transitions)
                next_output += args.output_period
                last_output_ep = len(rewards)

    with open(os.path.join(args.ckpt_dir, args.train_record_fname), 'a') as f:
        f.write('\n\n')

def show_game(args):
    env = gym.make('BreakoutDeterministic-v4')
    tf.reset_default_graph()
    qnet = get_qnet(args)

    saver = tf.train.Saver()
    tf.get_default_graph().finalize()
    with tf.Session() as sess:
        saver.restore(sess, args.restore_ckpt)
        done = False
        img = env.reset()
        _ = env.render()
        img = preprocess_img(img)
        state = np.stack((img, img, img, img), axis=2)
        reward, turns = 0, 0

        while not done:
            t1 = time.time()
            action = qnet.predict(sess, normalize(np.array([state])))[0]
            if args.show_random and np.random.rand(1) < args.e_f:
                action = 1 # Doesn't seem to like restarting

            img, r, done, _ = env.step(action)
            _ = env.render()
            img = np.reshape(preprocess_img(img), (85, 80, 1))
            state = np.concatenate((state[:, :, 1:], img), axis=2)
            reward += r
            turns += 1
            time.sleep(max(0, .025 - (time.time() - t1)))

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


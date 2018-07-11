"""
This is a reimplementation of how Deepmind beat Atari games. Based on:
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
"""

import gym, os
import numpy as np
import tensorflow as tf
from RingBuf import ExpBuf
from datetime import datetime
from random import randint
from resource import getrusage, RUSAGE_SELF

xavier = tf.contrib.layers.xavier_initializer
E = 1 - 1e-5
CKPT_DIR = "/home/matan/PycharmProjects/BreakoutDQN/deepmind_ckpts"

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

def transform_reward(reward):
    return np.sign(reward)

def feed_states(states):
    """
    Prepares a list of states to be fed to the Network. Normalizes and returns
    as an np.array. Only to be called right before feeding.

    :param states:
    :return:
    """
    return normalize(np.array(states))


def fit_batch(model, gamma, start_states, actions, rewards, next_states,
              is_terminal):
    """
    Do one DQN learning iteration

    :param model: The DQN
    :param gamma: Discount factor for future rewards.
    :param start_states: np array of starting states.
    :param actions: np array of one-hot encoded actions for start_states.
    :param rewards: np array of rewards for start_states action actions.
    :param next_states: np array of states resulting from start_state + action.
    :param is_terminal: np array of bools for if state is end of game.
    :return:
    """
    # Predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q = model.predict([next_states, np.ones(actions.shape)])
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q = rewards + gamma * np.max(next_Q, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    model.fit(
        [start_states, actions], actions * Q[:, None],
        nb_epoch=1, batch_size=len(start_states), verbose=0
    )

def make_cnn(n_actions, states_input):
    """
    Makes a NN to take in a batch of states (4 of the preprocessed images) with
    an output of size n_actions. No activation function is applied to the final
    output.
    :param n_actions: env.action_space.n - 1
    :param states_input: normalized preprocessed batch of states (image sets)
    :return:
        - CNN from images to expected value
        - A model that gives a batch of predicted actions for a batch of
          normalized states.
    """
    conv1 = tf.layers.conv2d(states_input, 16, (8, 8), (4, 4),
                             activation=tf.nn.relu,
                             kernel_initializer=xavier(),
                             bias_initializer=xavier())
    conv2 = tf.layers.conv2d(conv1, 32, (4, 4), (2, 2),
                             activation=tf.nn.relu,
                             kernel_initializer=xavier(),
                             bias_initializer=xavier())
    hidden = tf.layers.dense(tf.layers.flatten(conv2), 256,
                             activation=tf.nn.relu,
                             kernel_initializer=xavier(),
                             bias_initializer=xavier())
    model = tf.layers.dense(hidden, n_actions,
                           kernel_initializer=xavier(),
                           bias_initializer=xavier())
    predictions = tf.argmax(model, 1)
    return model, predictions

def make_train_op(n_actions, model):
    """
    Defines a training operation, along with the necessary inputs, to that the
    NN can learn.
    :param n_actions: env.action_space.n - 1
    :param model: neural network used to predict actions from states/imgs.
    :return:
    """
    # Create a one_hot encoding of the actions taken as a way to isolate the
    # correct Q value of the action taken in a given state.
    actions_taken = tf.placeholder(shape=(None), dtype=tf.int32)
    actions_onehot = tf.one_hot(actions_taken, n_actions, dtype=tf.float32)
    vals = tf.reduce_sum(model * actions_onehot, axis=1)

    # Take the 'true' Q value to calculate error. Q* = y * Q(s', a') + r
    target_vals = tf.placeholder(shape=(None), dtype=tf.float32)

    # Train by reducing the MSE via Gradient Descent
    loss = tf.reduce_mean(tf.square(target_vals - vals))
    optimizer = tf.train.RMSPropOptimizer(.00025, decay=.95, epsilon=.01)
    train_op = optimizer.minimize(loss)

    return actions_taken, target_vals, train_op

def update_model(sess, experiences, states_input, model, predictions,
                 actions_input, target_vals_input, train_op):
    """
    Train the model. Takes in a batch of experiences to learn from.
    :param sess: tf.Session
    :param experiences: list of Experiences from the ExpBuf.
    :param states_input: tf.placeholder, input to network
    :param model: NN that gives expected value for each action given a state/batch.
    :param predictions: Actions the model would select for each state in a batch.
    :param actions_input: tf.placeholder for the actions taken when a state was first encountered.
    :param target_vals_input: tf.placeholder for "true" value of a state. Used to compare the expected value produced by model.
    :param train_op: tf.train.optimizer, SGD learner
    :return:
    """
    # Break apart the experiences and normalize the states
    states = feed_states([e.state for e in experiences])
    actions = [e.action for e in experiences]
    rewards = [e.reward for e in experiences]
    next_states = feed_states([e.next_state for e in experiences])
    not_terminal = [not e.is_terminal for e in experiences]

    # Predict what actions should be taken in the next_state so we can
    # select the appropriate nextQ to calculate to discounted future value.
    next_actions = sess.run(predictions, feed_dict={states_input: next_states})
    rawQ = sess.run(model, feed_dict={states_input: next_states})
    nextQ = rawQ[:, next_actions]

    # discounted future value ('true' Q) = r +
    target_vals = not_terminal * (rewards + .99 * nextQ)

    # Update the model based on knowing the true Q value
    _ = sess.run(train_op,
                 feed_dict = {
                     states_input: normalize(states),
                     actions_input: actions,
                     target_vals_input: target_vals})


    #state, done, reward = play_turn(state, states_input, env, e, exp_buf)
def play_turn(sess, state, states_input, predictions, env, e, exp_buf, lives,
              n_actions):
    norm_state = feed_states([state])
    action = sess.run(
        predictions, feed_dict={states_input: norm_state})[0]
    if np.random.rand(1) < e:
        action = randint(0, n_actions-1)

    # Perform the action and process the results for storage
    # - preprocess the img and create the new state
    # - clip the reward and account for 'dying'
    # - store the experience
    next_raw_img, r, done, info = env.step(action + 1)
    next_img = np.reshape(preprocess_img(next_raw_img), (105, 80, 1))
    next_state = np.concatenate((state[:, :, :3], next_img), axis=2)
    r = np.clip(r, -1, 1) + lives - info['ale.lives']
    exp_buf.append(state, action, r, next_state, done)

    return next_state, done, r, info['ale.lives']


def main():
    exp_buf = ExpBuf(3e5) # ~10GB
    env = gym.make('BreakoutDeterministic-v4')
    tf.reset_default_graph()
    n_actions = 3

    states_input = tf.placeholder(shape=(None, 105, 80, 4), dtype=tf.float32)
    model, predictions = make_cnn(n_actions, states_input)
    actions_input, target_vals_input, train_op = make_train_op(n_actions, model)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        e = E
        episode = 0
        rewards = []
        turns = 0

        while True:
            done = False
            img = preprocess_img(env.reset())
            state = np.stack((img, img, img, img), axis=2)
            # Per game counters
            ep_turn, ep_reward, lives = 0, 0, 5

            while not done:
                state, done, r, lives = play_turn(
                    sess, state, states_input, predictions, env, e, exp_buf,
                    lives, n_actions)

                ep_turn += 1
                ep_reward += r

                if turns > 5e5:
                    # Once we have enough experiences in the buffer we can
                    # start learning.
                    update_model(sess, exp_buf.sample(8), states_input, model,
                                 predictions, actions_input, target_vals_input,
                                 train_op)
                    e *= E

                if done:
                    turns += ep_turn
                    rewards.append(ep_reward)

            episode += 1
            T = 1000 # periods of prints.
            if (episode + 1) % T == 0:
                buf_size_str = '' if exp_buf.size == exp_buf.capacity else \
                    ' exp_buf.size=' + str(exp_buf.size)
                e_str = '' if e < 1e-5 else ' e={:0.2e}'.format(e)
                mem_usg_str = ' mem_usage={:0.2f}GB'.format(
                    getrusage(RUSAGE_SELF).ru_maxrss / 2**20)
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S "),
                      mem_usg_str, ' episode=', episode+1,
                      ' reward_last_' + str(T) + '_games=', sum(rewards[-T:]),
                      buf_size_str, e_str, sep='')
                model_name = '-'.join(['deepmind', str(episode+1)])
                saver.save( sess, os.path.join(CKPT_DIR, model_name))

if __name__ == '__main__':
    main()

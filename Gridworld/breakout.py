import gym, time, random
import gridworld
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

LEARN_RATE = .1
EPISODES = 10000

def prep_image(image):
    # convert from raw atari game to input for NN
    grayscale = np.average(image, axis=2)
    normalized = grayscale / 255.
    return normalized.astype(np.float32)

def make_nn(input_layer, n_actions):
    # Expects a grayscale image as input
    input_shape = input_layer.shape
    reshaped_input = tf.reshape(
            input_layer,
            shape=(-1, input_shape[0], input_shape[1], 1))
    conv1 = tf.layers.conv2d(
        inputs=reshaped_input,
        filters=32,
        kernel_size=[6, 6],
        strides=[3, 3],
        padding="valid",
        activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='valid',
        activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[4, 4],
        strides=[1, 1],
        padding='valid',
        activation=tf.nn.relu)
    print(conv3.shape)
    dense1 = tf.layers.dense(
        inputs=tf.reshape(conv3, [-1, np.prod(conv3.shape[1:])]),
        units=1024,
        activation=tf.nn.sigmoid)
    return tf.layers.dense(inputs=dense1, units=n_actions)


def main():
    env = gym.make('Breakout-v0')
    tf.reset_default_graph()
    # Make grayscale so faster
    input_shape = env.observation_space.shape[:2]
    # Action 1 is just used to start the game and doesn't
    # move the thing. Action 0 does nothing.
    # Therefore I only want it to train on 3 actions and I
    # will manually start the game every time.
    n_actions = env.action_space.n - 1
    y = .99
    e = .1

    # Describe the NN used to select actions.
    input_layer = tf.placeholder(shape=input_shape, dtype=tf.float32)
    actions = make_nn(input_layer, n_actions)

    # Create learning parameters (loss)
    next_actions = tf.placeholder(shape=[1, n_actions], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_actions - actions))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE).minimize(loss)

    rewards = []
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(EPISODES):
            env.reset() # specific state
            net_reward = 0
            done = False
            lives = 5 # always starts with 5
            start_time = time.time()
            j = 0
            # start the game so don't waste time doing nothing
            s, _, _, _ = env.step(1)
            s = prep_image(s) 
            while not done:
                j += 1
                # Choose the action greedily but with chance e of random action
                # an action function (action) becomes a specific action (a)
                # a set of action functions (actions) becomes a set of Q values (Q)
                a, Q = sess.run(
                        [tf.argmax(actions, 1), actions],
                        feed_dict={input_layer: s})
                if np.random.rand(1) < e:
                    a[0] = random.randint(0, n_actions-1)
                # Take action and get the next state and reward
                next_s, reward, done, info = env.step(a[0] + 1)
                next_s = prep_image(next_s)
                # Try to speed up training (especially at beginning)
                # by giving more feedback
                reward -= lives - info['ale.lives']
                # Obtain the next Q values by feeding the new state to the network
                # and use it to train.
                next_Q = sess.run(actions, feed_dict={input_layer: next_s})
                Q[0, a[0]] = reward + y*np.max(next_Q)
                sess.run(train_op,
                         feed_dict={input_layer: s, next_actions: Q})
                net_reward += reward
                s = next_s

                if i % 10 == 0:
                    env.render()
                    #time.sleep(.01) # unneeded cuz my computer is slow :(
                if info['ale.lives'] < lives:
                    # start the next round so don't waste time doing nothing
                    env.step(1)
                    lives = info['ale.lives']
                if done:
                    # Reduce change of random action
                    e = 1. / ((1./50) + 10)
                    break
            print(i, ' - ', int(time.time() - start_time), ' - ', net_reward, ' - ', j)
            if i % 10 == 0:
                env.close()
            rewards.append(net_reward)

    plt.plot(rewards)
    plt.show()


if __name__ == '__main__':
    main()

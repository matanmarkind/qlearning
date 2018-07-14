import gridworld
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

LEARN_RATE = .1
EPISODES = 10001
DISCOUNT = .95
model_dir = '/home/matan/PycharmProjects/q-learning/'

def maybe_show(episode, img):
    if episode % 1000 == 0:
        plt.imshow(img)
        plt.draw()
        plt.pause(.05)

def make_nn(input_layer):
    """
    NN should be made with knowledge of the inputs and outputs, so not going
    to take them as parameters
    """
    reshaped_input = tf.reshape(
            input_layer,
            shape=(-1, 84, 84, 3))
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
    dense1 = tf.layers.dense(
        inputs=tf.reshape(conv3, [-1, np.prod(conv3.shape[1:])]),
        units=1024,
        activation=tf.nn.sigmoid)
    return tf.layers.dense(inputs=dense1, units=4)

def main():
    env = gridworld.gameEnv(partial=False, size=5)
    input_shape = env.state.shape
    tf.reset_default_graph()
    e = .5

    # Describe the NN used to select actions
    input_layer = tf.placeholder(shape=input_shape, dtype=tf.float32)
    actions = make_nn(input_layer)

    # Create learning parameters (loss)
    next_actions = tf.placeholder(shape=(1, env.actions), dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_actions - actions))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE).minimize(loss)

    rewards = []
    
    plt.ion()
    plt.show()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        try:
            saver.restore(sess, model_dir)
        except:
            print('New Model')

        for i in range(EPISODES):
            s = env.reset() # specific state
            net_reward = 0
            done = False
            j = 0
            while not done and j < 200:
                # Choose the action greedily but with chance e of random action
                # an action function (action) becomes a specific action (a)
                # a set of action functions (actions) becomes a set of Q values (Q)
                Q = sess.run(actions, feed_dict={input_layer: s})
                a = np.argmax(Q, 1)[0]
                if np.random.rand(1) < e:
                    a = np.random.randint(0, env.actions)
                # Perform action and get the next state and reward
                next_s, reward, done = env.step(a)
                maybe_show(i, s)
                # Obtain the next Q values by feeding the new state to the network
                # and use it to train.
                next_Q = sess.run(actions, feed_dict={input_layer: next_s})
                Q[0, a] = reward + DISCOUNT*np.max(next_Q)
                sess.run(train_op,
                         feed_dict={input_layer: s, next_actions: Q})
                net_reward += reward
                s = next_s
                j += 1

            e *= .99
            rewards.append(net_reward)
            if i % 1000 == 0 :
                plt.close()
                if i >= 1000:
                    print('Round', i, '- Net Reward last 100:',
                          sum(rewards[-1000:]))
        save_path = saver.save(sess, model_dir)
        print(save_path)

    plt.close()
    plt.ioff()
    plt.plot(rewards)
    plt.show()


if __name__ == '__main__':
    main()

import gym, time, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

LEARN_RATE = .01
EPISODES = 20001

def maybe_show(episode, env):
    if episode % 1000 == 0:
        print('----    ', episode, '    ----')
        _ = env.render()
        time.sleep(.1)

def make_nn(states, n_states, n_actions):
    """
    states: inputs, tf.placeholder
    """
    W = tf.Variable(tf.random_uniform([n_states, n_actions], 0, 0.1))
    return tf.matmul(states, W)

def get_model_fn(n_states, n_actions):
    def model_fn(features, labels, mode):
        """
        features: dictionary of inputs. (x from tf.estimator.inputs.numpy_input_fn)
        labels: expected outputs. (y from tf.estimator.inputs.numpy_input_fn)
        mode: train, eval, predict
        """
        actions = make_nn(features['states'], n_states, n_actions)
        predictions = {'action': tf.argmax(input=actions, axis=1)}

        # Create learning parameters (loss)
        next_actions = tf.placeholder(shape=[1, n_actions], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(next_actions - actions))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=LEARN_RATE)
            train_op = optimizer.minimize(loss)
            return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, train_op=train_op)

        except("Not ready for other modes yet.")
    return model_fn



def main():
    env = gym.make('FrozenLake-v0')
    tf.reset_default_graph()
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    y = .99
    e = .1

    # Describe the NN used to select actions
    states = tf.placeholder(shape=[1, n_states], dtype=tf.float32)
    actions = make_nn(states, n_states, n_actions)
    action = tf.argmax(actions, 1)

    # Create learning parameters (loss)
    next_actions = tf.placeholder(shape=[1, n_actions], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_actions - actions))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE).minimize(loss)

    rewards = []
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(EPISODES):
            s = env.reset() # specific state
            net_reward = 0
            done = False
            for j in range(100):
                # Choose the action greedily but with chance e of random action
                # an action function (action) becomes a specific action (a)
                # a set of action functions (actions) becomes a set of Q values (Q)
                a, Q = sess.run(
                        [action, actions],
                        feed_dict={states: np.identity(n_states)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                # Take action and get the next state and reward
                next_s, reward, done, _ = env.step(a[0])
                #maybe_show(i, env)
                # Obtain the next Q values by feeding the new state to the network
                # and use it to train.
                next_Q = sess.run(actions,
                                  feed_dict={states:np.identity(n_states)[next_s:next_s+1]})
                Q[0, a[0]] = reward + y*np.max(next_Q)
                sess.run(train_op,
                         feed_dict={states: np.identity(n_states)[s:s+1],
                                    next_actions: Q})
                net_reward += reward
                s = next_s
                if done:
                    # Reduce change of random action
                    e = 1. / ((1./50) + 10)
                    break
            rewards.append(net_reward)
            if not i == 0 and i % 1000 == 0:
                print('Successes in last 1000: ', sum(rewards[-1000:]))

    print('Percent of succesfull episodes: ', sum(rewards) / EPISODES)


if __name__ == '__main__':
    main()

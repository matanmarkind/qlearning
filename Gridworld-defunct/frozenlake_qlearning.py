import gym, time
import numpy as np

LEARN_RATE = .8
y = .95
EPISODES = 2001

def maybe_show(episode, env):
    if episode % 1000 == 0:
        print('----    ', episode, '    ----')
        _ = env.render()
        time.sleep(.1)
    

def main():
    # Use the OpenAI Gym Frozen Lake environment
    env = gym.make('FrozenLake-v0')
    # Initialize table of (states, actions) as 0
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    rewards = []
    for i in range(EPISODES):
        state = env.reset()
        net_reward = 0
        done = False
        for _ in range(100):
            # Given the state we are in, select an action.
            # Done greedily - we select the action with the greatest immediate
            # reward. Through training, each actions immediate reward should
            # be well correlated with long term reward.
            # Add noise, so will try new things (greater at beginning).
            action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i+1)))
            # Perform action and get the new state and reward
            state_next, reward, done, _ = env.step(action)
            #maybe_show(i, env)
            # Update Q-table
            Q[state, action] += LEARN_RATE * (reward + y*np.max(Q[state_next, :]) - Q[state, action])
            net_reward += reward
            state = state_next
            if done:
                break
        rewards.append(net_reward)

    print('Percent of successful episodes: ', sum(rewards) / EPISODES)
    print('Final Q-table:')
    print(Q)

if __name__ == '__main__':
    main()

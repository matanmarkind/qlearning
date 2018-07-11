from random import randint
import numpy as np

class RingBuf:
    def __init__(self, capacity):
        """
        Initialize an empty list of capacity elements to be filled up.
        :param capacity: Maximum number of elements to store before wrapping.
        """
        self.capacity = int(capacity)
        self.index = 0
        self.data = [None] * self.capacity

    def __getitem__(self, item):
        return self.data[item]

    def append(self, element):
        """
        Internal append that will be fronted by the specific buffer that will
        provide a nice interface. Add a new element to the list.
        Overwrites in the order that elements
        were placed on buffer.
        :param element: new element to append.
        :return:
        """
        self.data[self.index % self.capacity] = element
        self.index += 1

    def sample(self, num):
        """
        Randomly sample a set of the elements in the buffer.
        :param num: Number of elements to sample.
        :return: a list of elements.
        """
        ids = [randint(0, self.size-1) for i in range(num)]
        return [self.data[i] for i in ids]


    @property
    def size(self):
        """
        While the buffer is still growing self.index. Once the buffer is full
        the length stays stagnant at capacity while new elements simple overwrite
        the old ones.
        :return:
        """
        return min(self.index, self.capacity)

    def __len__(self):
        return self.size


class ExpBuf():
    def __init__(self, capacity):
        """
        An experience buffer to hold onto the experiences for a network. Each
        experience is held at the same index across each attribute.
        :param capacity:
        """
        self.state = RingBuf(capacity)
        self.action = RingBuf(capacity)
        self.reward = RingBuf(capacity)
        self.next_state = RingBuf(capacity)
        self.not_terminal = RingBuf(capacity)


    def append(self, state, action, reward, next_state, is_terminal):
        """
        Takes in an experience that a network had for later replay
        :param state: preprocessed image group
        :param action: action taken when first encountered state
        :param reward: reward received for action
        :param next_state: preprocessed state that resulted from the action
        :param is_terminal: did the game finish
        :return:
        """
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.not_terminal.append(not is_terminal)

    def sample(self, num):
        """
        Randomly sample a set of the experiences in the buffer.
        :param num: Number of elements to sample.
        :return: a tuple of experiences, with each elements being an np.array
            for each attribute.
        """
        ids = [randint(0, self.state.size-1) for i in range(num)]
        states = np.array([self.state[i] for i in ids])
        actions = np.array([self.action[i] for i in ids])
        rewards = np.array([self.reward[i] for i in ids])
        next_states = np.array([self.next_state[i] for i in ids])
        not_terminals = np.array([self.not_terminal[i] for i in ids])
        return states, actions, rewards, next_states, not_terminals

    @property
    def capacity(self):
        return self.state.capacity

    def __len__(self):
        return len(self.state)

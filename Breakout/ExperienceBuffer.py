from RingBuffer import RingBuf, WeightedRingBuf
import numpy as np

class ExpBuf():
    class Experience():
        def __init__(self, state, action, reward, next_state, not_terminal):
            self.state = state
            self.action = action
            self.reward = reward
            self.next_state = next_state
            self.not_terminal = not_terminal

    def __init__(self, capacity):
        """
        An experience buffer to hold onto the experiences for a network. Each
        experience is held at the same index across each attribute.
        :param capacity:
        """
        self.experiences = RingBuf(capacity)

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
        self.experiences.append(self.Experience(
            state, action, reward, next_state, not is_terminal))

    def sample(self, num):
        """
        Randomly sample a set of unique experiences from the buffer.
        :param num: Number of elements to sample.
        :return: a tuple of experiences, with each elements being an np.array
            for each attribute.
        """
        state, action, reward, next_state, not_terminal = [], [], [], [], []
        for exp in self.experiences.sample(num):
            state.append(exp.state)
            action.append(exp.action)
            reward.append(exp.reward)
            next_state.append(exp.next_state)
            not_terminal.append(exp.not_terminal)

        return np.asarray(state), np.asarray(action), np.asarray(reward), \
               np.asarray(next_state), np.asarray(not_terminal)

    @property
    def capacity(self):
        return self.experiences.capacity

    def __len__(self):
        return len(self.experiences)

class WeightedExpBuf():
    """
    An experience buffer to hold the memory for a neural network so that
    states the network experiences can be replayed. Weights the likelihood
    of selecting an experience based on its loss. To update these weights as
    the network replays experiences each memory will also come with an id
    so that the network can update the losses after replay.
    """
    class Experience():
        def __init__(self, state, action, reward, next_state, not_terminal, weight):
            self.state = state
            self.action = action
            self.reward = reward
            self.next_state = next_state
            self.not_terminal = not_terminal
            self.weight = weight

    def __init__(self, capacity):
        """
        A binary tree where each leaf's value is a weight which is used to
        determine the probability of selecting a given leaf.
        """
        self.experiences = WeightedRingBuf(capacity)

    def append(self, state, action, reward, next_state, is_terminal):
        """
        Takes in an experience that a network had for later replay. Initially
        each experience has a loss of 0, since we haven't trained on it yet,
        merely encountered it.

        :param state: preprocessed image group
        :param action: action taken when first encountered state
        :param reward: reward received for action
        :param next_state: preprocessed state that resulted from the action
        :param is_terminal: did the game finish
        :return:
        """
        self.experiences.append(self.Experience(
            state, action, reward, next_state, not is_terminal, 0))

    def sample(self, num):
        """
        Randomly sample a set of unique experiences from the buffer.
        :param num: Number of elements to sample.
        :return: a group of alligned vectors with the following:
            - ids: id of an experience (used to update the weights)
            - state
            - action when first encountered the state
            - reward when first encountered the state
            - next_state (as a result of taking action in state)
            - not_terminal (did the episode continue after this state?)
        """
        ids = self.experiences.sample(num)
        state, action, reward, next_state, not_terminal = [], [], [], [], []
        for exp in self.experiences[ids]:
            state.append(exp.state)
            # action.append(exp.action)
            reward.append(exp.reward)
            next_state.append(exp.next_state)
            not_terminal.append(exp.not_terminal)

        return ids, np.asarray(state), np.asarray(action), np.asarray(reward), \
               np.asarray(next_state), np.asarray(not_terminal)

    def update_weights(self, exp_ids, new_weights):
        """
        Take a batch of (memory_id, weight) with which to update the tree.
        """
        for idx, weight in zip(exp_ids, new_weights):
            self.experiences.update_weight(idx, weight)

    @property
    def capacity(self):
        return self.experiences.capacity

    def __len__(self):
        return len(self.experiences)

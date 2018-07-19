from collections import deque
import numpy as np
import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)
from utils.RingBuffer import RingBuf, WeightedRingBuf

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
        :return: index of element, element overwritten
        """
        exp = self.Experience(
            state, action, reward, next_state, not is_terminal)
        idx, old_ele = self.experiences.append(exp)
        return idx, old_ele


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
        # ids of experiences that haven't been used for training yet.
        self.unplayed_experiences = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, is_terminal, weight=0):
        """
        Takes in an experience that a network had for later replay. Initially
        each experience has a loss of 0 (unless given), since we haven't trained
        on it yet, merely encountered it. This is a trash value and expects
        to be updated when the experience is first replayed.

        :param state: preprocessed image group
        :param action: action taken when first encountered state
        :param reward: reward received for action
        :param next_state: preprocessed state that resulted from the action
        :param is_terminal: did the game finish
        :param weight: how much to weight the experience.
        :return: index of the appended element, element overwritten
        """
        idx, old_ele = self.experiences.append(self.Experience(
            state, action, reward, next_state, not is_terminal, weight))
        self.unplayed_experiences.append(idx)
        return idx, old_ele

    def sample(self, batch_size):
        """
        Select up to batch_size elements from the list of experiences
        that have never been used to learn from. Then fill out the
        sample from the weighted buffer by randomly sampling.

        :param batch_size: Number of elements to sample.
        :return: a group of alligned vectors with the following:
            - ids: id of an experience (used to update the weights)
            - state
            - action when first encountered the state
            - reward when first encountered the state
            - next_state (as a result of taking action in state)
            - not_terminal (did the episode continue after this state?)
        """
        ids = set()
        while len(ids) < batch_size and len(self.unplayed_experiences) > 0:
          # Make sure to replay new experiences before sampling. This
          # guarantees all experiences get 1 replay. It also allows
          # us to set the weights of these experiences, since they
          # will (probably) be weighted 0 when first appended.
          ids.add(self.unplayed_experiences.pop())

        # sample the from the weighted buffer, but make sure to exclude the ids
        # we give from the unplayed set.
        ids |= self.experiences.sample(batch_size - len(ids), exclude=ids)
        assert len(ids) == batch_size,\
            "Internal Error: incorrect sample size. len(ids)=" + str(len(ids))

        state, action, reward, next_state, not_terminal = [], [], [], [], []
        for exp in self.experiences[ids]:
            state.append(exp.state)
            action.append(exp.action)
            reward.append(exp.reward)
            next_state.append(exp.next_state)
            not_terminal.append(exp.not_terminal)

        return ids, np.asarray(state), np.asarray(action), np.asarray(reward), \
               np.asarray(next_state), np.asarray(not_terminal)

    def update_weights(self, exp_ids, new_weights):
        """
        Take a batch of (memory_id, weight) with which to update the tree.
        """
        assert len(set(exp_ids)) == len(exp_ids),\
            "Invalid Argument: must pass a unique set of experience ids."

        for idx, weight in zip(exp_ids, new_weights):
            self.experiences.update_weight(idx, weight)

    @property
    def capacity(self):
        return self.experiences.capacity

    def __len__(self):
        return len(self.experiences)

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

    Prioritization is done according to weights provided, this is in contrast
    to rank based.
    """
    class Experience():
        def __init__(self, state, action, reward, next_state, not_terminal, weight):
            self.state = state
            self.action = action
            self.reward = reward
            self.next_state = next_state
            self.not_terminal = not_terminal
            self.weight = weight  # (error + offset) ^ alpha

        def __str__(self):
          return '[{}, {}, {}, {}, {}]'.format(self.state, self.action,
                                               self.reward, self.next_state,
                                               self.not_terminal)

    def __init__(self, capacity, alpha, beta_i, beta_f, beta_anneal,
                 weight_offset):
        """
        A binary tree where each leaf's value is a weight which is used to
        determine the probability of selecting a given leaf. 

        :param capacity: number of experiences to hold in RingBuf
        :param alpha: How much to weight prioritization.
        :param beta_i: initial weighting for bias correction.
        :param beta_f: final weighting for bias correction.
        :param beta_anneal: number of updates over which to anneal to beta_f.
        :param weight_offset: small positive number to prevent 0 weighting.
            Also works to slightly smooth out the disribution.
        """
        self.weight_offset = weight_offset
        self.alpha = alpha

        assert beta_i < beta_f, "Beta update assumes beta_i < beta_f"
        self.beta = beta_i
        self.beta_f = beta_f
        self.beta_update = (beta_f - beta_i) / beta_anneal

        self.experiences = WeightedRingBuf(capacity)
        # ids of experiences that haven't been used for training yet.
        self.unplayed_experiences = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, is_terminal):
        """
        Takes in an experience that a network had for later replay.

        Initially the experience is given 0 weight. It will be guaranteed
        replay though before any experience that has already been replayd,
        and then the user can give the experience a valid weight.

        :param state: preprocessed image group
        :param action: action taken when first encountered state
        :param reward: reward received for action
        :param next_state: preprocessed state that resulted from the action
        :param is_terminal: did the game finish
        :return: index of the appended element, element overwritten
        """
        idx, old_ele = self.experiences.append(self.Experience(
            state, action, reward, next_state, not is_terminal, 0))
        self.unplayed_experiences.append(idx)
        return idx, old_ele

    def sample(self, batch_size):
        """
        Sample batch_size experiences (from the last capacity added).
        Select experiences that have yet to be replayed, and fill in the
        remaining elements by sampling from the buffer.

        :param batch_size: Number of elements to sample.
        :return: a group of alligned vectors (ith element of each vector comes
                 from the same experience) each of len batch_size with the
                 following:
            - ids: id of an experience (used to update the weights)
            - state
            - action when first encountered the state
            - reward when first encountered the state
            - next_state (as a result of taking action in state)
            - not_terminal (did the episode continue after this state?)
            - Importance Sampling weights, used to correct for bias.
        """
        max_IS_weight = 0
        assert len(self) >= batch_size, "Can't sample " + str(batch_size) +\
            " unique elements if there aren't that many elements."
        ids = set()
        IS_weights = dict()
        while len(ids) < batch_size and len(self.unplayed_experiences) > 0:
          # Make sure to replay new experiences before sampling. This
          # guarantees all experiences get 1 replay. Which is important so
          # that the network gets a variety of experiences, and also so that
          # the experience can be given a legitimate weight.
          idx = self.unplayed_experiences.popleft()
          ids.add(idx)
          # On first replay the experience is of full importance.
          IS_weights[idx] = 1

        if len(ids) < batch_size:
          assert self.total_weight > 0,\
              "Attempting to re-sample without updating weights."
          # Precompute for Importance Sampling.
          P_min = (self.min_weight / self.total_weight)
          max_IS_weight = (self.capacity * P_min) ** -self.beta

          # sample from the weighted buffer, but make sure to exclude the ids
          # that were chosen from the unplayed ones.  Break the tree up into
          # ranges of weights so that we sample from a range of different
          # episodes.
          ids |= self.experiences.sample_n_subsets(batch_size - len(ids),
                                                   exclude=ids)
          assert len(ids) == batch_size,\
              "Internal Error: incorrect sample size. len(ids)=" + str(len(ids))

        state, action, reward, next_state, not_terminal = [], [], [], [], []
        for idx, exp in zip(ids, self.experiences[ids]):
            state.append(exp.state)
            action.append(exp.action)
            reward.append(exp.reward)
            next_state.append(exp.next_state)
            not_terminal.append(exp.not_terminal)
            if idx not in IS_weights.keys():
                # For previously experienced weights, calculate and append
                # the Importance Sampling weight.
                P_i = exp.weight / self.total_weight
                weight_i = (self.capacity * P_i) ** -self.beta
                IS_weights[idx] = weight_i / max_IS_weight

        return ids, np.array(state), np.array(action), np.array(reward), \
               np.array(next_state), np.array(not_terminal),\
               np.array([IS_weights[idx] for idx in ids])

    def update_weights(self, exp_ids, raw_weights):
        """
        Update the weighting of experiences for prioritized sampling.
        Internally apply exponential priority importance function
        (raw_weight + epsilon) ^ alpha.

        :param exp_ids: iterable of the indices of the transitions to update
        :param raw_weights: np.array of raw weight value for the matching
            experience. (likely TD error)

        Expects the loss to always be positive.
        """
        assert len(set(exp_ids)) == len(exp_ids),\
            "Invalid Argument: must pass a unique set of experience ids."

        new_weights = (raw_weights + self.weight_offset) ** self.alpha

        # Update the weights used for sampling each of the experiences.
        for idx, weight in zip(exp_ids, new_weights):
            self.experiences.update_weight(idx, weight)

        # Update beta which is used to weight the importance sampling.
        if self.beta < self.beta_f:
            self.beta = min(self.beta_f, self.beta + self.beta_update)

    @property
    def capacity(self):
        return self.experiences.capacity

    @property
    def total_weight(self):
        """
        The total of all the probabilistic weights in the sum tree.
        sum((raw_weight_i + weight_offset) ** alpha).
        """
        return self.experiences.total_weight

    @property
    def min_weight(self):
        """
        Minimum weight value of all elements.
        """
        return self.experiences.min_weight

    def __len__(self):
        return len(self.experiences)

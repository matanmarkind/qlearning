from random import sample
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

    def __getitem__(self, items):
        """
        Returns the items inside of the buffer. Handles wraparound.
        """
        if isinstance(items, list):
            return [self.data[idx % self.capacity] for idx in items]
        return self.data[items % self.capacity]

    def append(self, element):
        """
        Internal append that will be fronted by the specific buffer that will
        provide a nice interface. Add a new element to the list.
        Overwrites in the order that elements
        were placed on buffer.
        :param element: new element to append.
        :return: the element replaced if wrapping around or None.
        """
        idx = self.index % self.capacity
        old_ele = self.data[idx]
        self.data[idx] = element
        self.index += 1
        return old_ele

    def sample(self, num):
        """
        Randomly sample a set of the elements in the buffer.
        :param num: Number of elements to sample.
        :return: a list of elements.
        """
        ids = sample(range(self.size), num)
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


class WeightedRingBuf():
    """
    A ring buffer which holds weighted elements. The weights determine the
    probability of selecting a given item when sampling. Can hold any element
    so long as it has a .weight property which is numeric.

    Implemented as a binary tree where each node holds the sum of the weights
    of its children.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.index = 0
        self.tree = self.make_tree(capacity)

    def __getitem__(self, idx):
        """
        Retrieve an item from the underlying buffer.
        :param idx: idx of the element.
        :return:
        """
        return self.tree[-1][idx]

    def append(self, ele):
        """
        Appends an element to the buffer and updates the weights of its
        parent nodes.
        :param ele:
        :return: old element
        """
        weight = ele.weight
        ele.weight = 0
        old_ele = self.tree[-1].append(ele)
        old_weight = 0 if old_ele is None else old_ele.weight
        
        self._update_weight(self.index % self.capacity, weight - old_weight)
        # Must reset ele.weight since now it is (weight - old_weight)
        ele.weight = weight
        
        self.index += 1
        return old_ele

    def make_tree(self, capacity):
        """
        Create a tree with all weights initialized to 0.
        """
        c = 1
        tree = []
        while c < capacity:
            tree.append(np.zeros(c))
            c *= 2
        tree.append(RingBuf(capacity))
        return tree

    def get_leaf(self):
        """
        Randomly select a leaf from the tree based on the weights in the tree.
        Returns the id of the leaf (the index).
        """
        val = np.random.randint(0, self.tree[0][0])
        idx = 0
        for depth in range(1, len(self.tree) - 1):
            left_weight = self.tree[depth][idx]
            if val >= left_weight:
                val -= left_weight
                idx = (idx + 1) * 2
            else:
                idx *= 2
        left_weight = self[idx].weight
        return idx + (val >= left_weight)
    
    def update_weight(self, index, weight):
        """
        Reset the weight of an element and update the weights in the tree.
        """
        idx = index % self.capacity
        delta = weight - self[idx].weight
        self._update_weight(idx, delta)

    def _update_weight(self, idx, delta):
        """
        Resets the weight of the element identified by index, then
        goes up each row updating the parent nodes with the new weight.
        :param idx: index of the changed experience.
        :param delta: change in weight of the element.
        """
        self[idx].weight += delta
        idx //= 2
        for depth in range(-2, -len(self.tree) - 1, -1):
            self.tree[depth][idx] += delta
            idx //= 2

    def sample(self, num):
        """
        Sample a number of unique experiences. Shouldn't cause too much change
        in the effective weights since we are assuming the batch_size is much
        smaller than the experience buffer.
        :param num: how many experiences to sample.
        Returns a unique set of indices for the leaves in the tree.
        """
        idxs = set()
        while len(idxs) < num:
            idxs.add(self.get_leaf())
        return list(idxs)

    
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
        self.exp_count = 0  # num experiences added
        self.capacity = capacity
        self.tree = self.make_tree(capacity)

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
        exp = Experience(state, action, reward, next_state, not is_terminal, 0)


    def update_tree(self, new_weights):
        """
        Take a batch of (memory_id, weight) with which to update the tree.
        """
        for index, weight in new_weights:
            delta = weight - self.tree[-1][index].weight
            self._update_tree(index, delta)



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
    
class WeightedExpBuf():
    def __init__(self, capacity):
        elf.state = RingBuf(capacity)
        self.action = RingBuf(capacity)
        self.reward = RingBuf(capacity)
        self.next_state = RingBuf(capacity)
        self.not_terminal = RingBuf(capacity)

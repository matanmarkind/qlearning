import numpy as np
import random

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
        ids = random.sample(range(len(self)), num)
        return [self.data[i] for i in ids]

    def __len__(self):
        """
        While the buffer is still growing self.index. Once the buffer is full
        the length stays stagnant at capacity while new elements simple overwrite
        the old ones.
        :return:
        """
        return min(self.index, self.capacity)


class WeightedRingBuf():
    """
    A ring buffer which holds weighted elements. The weights determine the
    probability of selecting a given item when sampling. Can hold any element
    so long as it has a .weight property which is numeric and writeable.

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
        val = random.uniform(0, self.tree[0][0])
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
        Sample a number of unique experiences. The uniqueness criteria shouldn't
        cause too much change in the effective weights since we are assuming the
        batch_size is much smaller than the experience buffer.
        :param num: how many experiences to sample.
        Returns a unique set of indices for the leaves in the tree.
        """
        idxs = set()
        while len(idxs) < num:
            idxs.add(self.get_leaf())
        return list(idxs)

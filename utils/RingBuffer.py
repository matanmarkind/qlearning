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
        self.iter_index = 0

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
        return idx, old_ele

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

    def __getitem__(self, items):
        """
        Returns the items inside of the buffer. Handles wraparound.
        """
        if isinstance(items, list) or isinstance(items, set) or isinstance(items, tuple):
            return [self.data[idx % self.capacity] for idx in items]
        elif isinstance(items, slice):
            return self.data[items]
        return self.data[items % self.capacity]

    def __iter__(self):
        """
        Allow user to iterate over the elements in the buffer.
        """
        return self

    def __next__(self):
        """
        Check if there are any valid elements left to return. If not, either
        have reached the last element in the buffer or have reached None (aka
        element that has never been set) raise StopIteration and reset the
        iteration counter.

        :return: single element from the buffer
        """
        if self.iter_index >= self.capacity or self[self.iter_index] is None:
            self.iter_index = 0
            raise StopIteration
        else:
            self.iter_index += 1
            return self[self.iter_index - 1]


class WeightedRingBuf():
    """
    A ring buffer which holds weighted elements. The weights determine the
    probability of selecting a given item when sampling. Can hold any element
    so long as it has a .weight property which is numeric and writeable.

    Implemented as a binary tree where each node holds the sum of the weights
    of its children.

    Elements don't even need to be all the same so long as all have weight
    property which returns a positive number.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = self.make_tree(capacity)
        self.min_weight = float("inf")

    @staticmethod
    def make_tree(capacity):
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

    def append(self, ele):
        """
        Appends an element to the buffer and updates the weights of its
        parent nodes. If the weight of the element is 0, will not update
        min_weight.
        :param ele: object with weight property (ele.weight returns a number).
        :return: old element
        """
        weight = ele.weight
        ele.weight = 0
        idx, old_ele = self.ring_buf.append(ele)
        old_weight = 0 if old_ele is None else old_ele.weight
        
        self._update_weight(idx, weight - old_weight)
        # Must reset ele.weight since now it is (weight - old_weight)
        ele.weight = weight

        self._update_min_weight(old_weight, weight)
        return idx, old_ele

    def update_weight(self, index, weight):
        """
        Reset the weight of an element and update the weights in the tree.
        """
        idx = index % self.capacity
        old_weight = self[idx].weight
        delta = weight - old_weight
        self._update_weight(idx, delta)
        self._update_min_weight(old_weight, weight)

    def sample(self, n, exclude=set()):
        """
        Sample n unique experiences. The uniqueness criteria shouldn't
        cause too much change in the effective weights since we are assuming the
        batch_size is much smaller than the experience buffer and that there
        aren't any really extreme outliers that represent a significant
        portion of the total_weight (I'm gunna ballpark and say so long as
        no element is > total_weight/n we should be ok).

        :param n: how many experiences to sample.
        :param exclude: excluce certain elements, handles wraparound.

        Returns a unique set of indices for the leaves in the tree.
        """
        exclude_idx = set([index % self.capacity for index in exclude])
        idxs = set()
        while len(idxs) < n:
            idx = self._sample(exclude_idx)
            exclude_idx.add(idx)
            idxs.add(idx)
        return idxs

    def sample_n_subsets(self, n, exclude=set()):
        """
        Sample n experiences each from a different subsection of the tree
        representing 1/n of the tree by weight (instead of 1/n of the elements)
        """
        if n == 0:
            return set()

        exclude_idx = set([index % self.capacity for index in exclude])
        idxs = set()
        min_weight = 0
        weight_step = self.total_weight / n

        for i in range(n):
            max_weight = min(min_weight + weight_step, self.total_weight)
            idx = self._sample(exclude_idx,
                               min_weight=min_weight,
                               max_weight=max_weight)
            exclude_idx.add(idx)
            idxs.add(idx)
            min_weight += weight_step
        return idxs

    @property
    def total_weight(self):
        return self.tree[0][0]

    @property
    def ring_buf(self):
        return self.tree[-1]

    def _sample(self, exclude_idx=set(), min_weight=0.0, max_weight=None):
        """
        Sample a single element which falls within a subset of the trees
        weights and is has not been requested to exclude.
        :param exclude_idx: modded index of elements not to sample.
        :param min_weight: used to set a minimum weight if sammpling from a
            subsection of the tree.
        :param max_weight: used to set a maximum weight if sammpling from a
            subsection of the tree.
        """
        while True:
            leaf = self._get_leaf(min_weight, max_weight)
            if leaf not in exclude_idx:
                return leaf

    def _get_leaf(self, min_weight=0, max_weight=None):
        """
        Randomly select a leaf from the tree based on the weights in the tree.
        Returns the id of the leaf (the index).

        :param min_weight: minimum weight to sample from.
        :param max_weight: maximum weight to sample from.
        """
        max_weight = self.total_weight if max_weight is None else max_weight
        assert min_weight >= 0.0 and min_weight <= max_weight and \
                max_weight <= self.total_weight,\
                "Invalid weight subsetting: min_weight=" + str(min_weight) +\
                ' max_weight=' + str(max_weight) + " total_weight=" +\
                str(self.total_weight)

        val = random.uniform(min_weight, max_weight)
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

    def _update_min_weight(self, old_weight, weight):
        """
        Updates the minimum weight for the buffer. Used whenever an
        element has its weight updated or is replaced. 

        If old weight is 0, guarantees not to trigger a find()
        """
        if weight == 0:
            # A weight of 0 is an indicator of an invalid element since it is
            # unreachable via sampling.
            return

        if weight < self.min_weight:
            self.min_weight = weight
        elif old_weight == self.min_weight:
            # If the previous min_weight was replaced, and the new weight
            # isn't the replacement, we must search all elements for the
            # new minimum weight.
            self.min_weight = self._find_min_weight()

    def _find_min_weight(self):
        """
        Find the element with the minimum weight in the buffer and return
        its weight. This ignores elements with 0 weight, since they
        are unreachable via sampling.

        O(capacity), but given that sampling is done in proportion to the
        weights, this should be called quite rarely.
        """
        min_weight = float("inf")
        for ele in self.ring_buf:
            if ele.weight != 0 and ele.weight < min_weight:
                min_weight = ele.weight
        return min_weight

    def __len__(self):
        """
        The tree is here just to manage the weights. The elements themselves
        sit at the base of the tree in a RingBuf. That is what tells us the
        effective size/number of elements.
        """
        return len(self.ring_buf)

    def __getitem__(self, idx):
        """
        Retrieve an item from the underlying buffer.
        :param idx: idx of the element.
        :return:
        """
        return self.ring_buf[idx]

    def __iter__(self):
        """
        Allow user to iterate over the elements in the buffer.
        """
        return self.ring_buf.__iter__()

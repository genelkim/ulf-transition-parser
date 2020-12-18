"""
 Represents a partial or complete dependency parse of a sentence, and
 provides convenience methods for analyzing the parse.
"""
from .utils import NULL
class DependencyTree(object):
    def __init__(self, tree=None):
        self.n = 0
        if tree:
            self.n = tree.n
            self.head_list = tree.head_list[:]
            self.label_list = tree.label_list[:]
        else:
            self.head_list = []
            self.label_list = []
        self.dist_stats = {}
        self.root = -1
        self.dist_threshold = 4
        self._ROOT_LABEL = "%^ROOT^%"

    def add(self, h, l):
        """
        Add the next token to the parse.
        :param h: the head index for next token.
        :param l: the dependency label between this token and its head.
        :return: None
        """
        self.n += 1
        self.head_list.append(h)
        self.label_list.append(l)

    # build dependence all the way with upper the dist upper bound
    def buildDepDist(self, upper):
        def pathLength(first_idx, second_idx):
            firstDists = {}
            firstDists[first_idx] = 0
            curr_idx = first_idx
            # upper is basically how deep along one line one searches
            for i in range(upper):
                # If element DNE
                if self.head_list[curr_idx] == -1:
                    break
                # head_list returns the head index for the NEXT token
                curr_idx = self.head_list[curr_idx]
                if curr_idx == second_idx:
                    return i+1
                # Loop over to get next token starting from first_index, understandable
                firstDists[curr_idx] = i+1

            curr_idx = second_idx
            # half-way, so if could connect, able to get dist
            for i in range(upper):
                if self.head_list[curr_idx] == -1:
                    return upper
                curr_idx = self.head_list[curr_idx]
                if curr_idx in firstDists:
                    curr_dist = firstDists[curr_idx] + i + 1
                    if curr_dist > upper:
                        curr_dist = upper
                    return curr_dist
            return upper

        for i in range(self.n):
            for j in range(i+1, self.n):
                curr_dist = pathLength(i, j)
                self.dist_stats[(i, j)] = curr_dist
                self.dist_stats[(j, i)] = curr_dist

    def getDepDist(self, first_idx, second_idx):
        return self.dist_stats[(first_idx, second_idx)]

    # if are directly connected
    def getDepLabel(self, first_idx, second_idx):
        if first_idx < 0 or second_idx < 0:
            return NULL
        if self.getHead(first_idx) == second_idx:
            return "L-" + self.getLabel(first_idx)
        if self.getHead(second_idx) == first_idx:
            return "R-" + self.getLabel(second_idx)
        return NULL

    # set the next token
    def setDependency(self, k, h, label):
        self.head_list[k] = h
        self.label_list[k] = label

    def getHead(self, k):
        if k < 0 or k >= self.n:
            return None
        return self.head_list[k]

    def getLabel(self, k):
        if k < 0 or k >= self.n:
            return None
        return self.label_list[k]

    # as the method name
    def getAllChildren(self, k):
        arcs = []
        for i in range(self.n):
            if self.getHead(i) == k:
                arcs.append(self.label_list[i])
        return arcs

    #?
    def getParent(self, k):
        return self.label_list[k]

    def getRoot(self):
        if self.root != -1:
            return self.root
        for k in range(self.n):
            if self.getHead(k) == -1:
                self.root = k
                return self.root
        return 0

    def isSingleRoot(self):
        n_root = 0
        for k in range(self.n):
            if self.getHead(k) == -1:
                n_root += 1
        return n_root == 1

    # if two trees are the same
    def __eq__(self, other):
        if other.n != self.n:
            return False
        for i in range(self.n):
            if self.getHead(i) != other.getHead(i):
                return False
            if self.getLabel(i) != other.getLabel(i):
                return False
        return True

    def getAdjacencyFieldInputs(self):
        """
        Generates a tuple of inputs for building a stog.data.fields.AdjacencyField.
        Adjacency field takes two parameters necessary from the dependency tree:
            indices: List[Tuple[int, int]] of edge maps
            labels: List[str] of edge labels
        For the purpose of representing the ROOT edges, we introduce a dummy
        ROOT self-edge.
        """
        return zip(*[
            ((head, i), label) if head != -1 else ((i, i), self._ROOT_LABEL)
            for i, head, label
            in zip(range(len(self.head_list)), self.head_list, self.label_list)
        ])


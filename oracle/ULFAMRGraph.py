from collections import defaultdict, deque
from .utils import NULL

class SymbolLabel(object):
    def __init__(self, label=None):
        self.value = label
        self.alignments = []
        self.rels = []
        self.rel_map = {}
        self.par_rel_map = {}

        self.tail_ids = []
        self.parent_rels = []
        self.parent_ids = []
        self.aligned = False
        self.isVar = False
        self.category = None
        self.map_info = None
        self.span = None

    def setVarType(self, v):
        self.isVar = v

    def setSpan(self, sp):
        self.span = sp

    def getRelStr(self, idx):
        assert idx in self.rel_map
        return self.rel_map[idx]

    def buildRelMap(self):
        n_rels = len(self.rels)
        for i in range(n_rels):
            curr_idx = self.tail_ids[i]
            self.rel_map[curr_idx] = self.rels[i]
        n_rels = len(self.parent_rels)
        for i in range(n_rels):
            curr_idx = self.parent_ids[i]
            self.par_rel_map[curr_idx] = self.parent_rels[i]

    def symbol_repr(self, graph):
        """Returns a contextualized string representation of this symbol the
        given graph. This representation includes this node, its outgoing arcs,
        and child values.
        """
        outgoing = []
        for (idx, l) in enumerate(self.rels):
            tail = self.tail_ids[idx]
            outgoing.append("%s:%s" % (l, graph.symbols[tail].value))
        return "%s %s" % (self.value, " ".join(sorted(outgoing)))


    def setValue(self, s):
        self.value = s

    def getValue(self):
        return self.value

    def rebuild_ops(self):
        """Rebuild the op relations."""
        op_tail_idxs = [tail_idx for (idx, tail_idx) in enumerate(self.tail_ids) if self.rels[idx] == "op"]
        if op_tail_idxs:
            sorted_ops = sorted(op_tail_idxs)
            idx_to_opstr = {}
            for (idx, tail_idx) in enumerate(sorted_ops, 1):
                idx_to_opstr[tail_idx] = "op%d" % idx
            for (idx, tail_idx) in enumerate(self.tail_ids):
                if tail_idx in idx_to_opstr:
                    self.rels[idx] = idx_to_opstr[tail_idx]

    def addAlignment(self, word_idx):
        self.alignments.append(word_idx)
        assert len(self.alignments) == 1

    def getArc(self, k):
        if k >= len(self.rels) or k < 0:
            return NULL
        return self.rels[k]

    def getParentArc(self, k):
        if k >= len(self.parent_rels) or k < 0:
            return NULL
        return self.parent_rels[k]

class ULFAMRGraph(object):
    """AMR Graph class with some alignment info.

    There are additional features for keeping track of restricted symbol indices,
    since these can be inconsisent between gold and hypothesis graphs.
    """
    def __init__(self, track_restricted=False):
        self.n = 0
        self.symbols = []
        self.counter = 0
        self.u_counter = 0
        self.widToSymbolID = {}
        self.symIDToWordID = {}
        self.sidToSpan = {}
        self.right_edges = defaultdict(list)
        self.toks = None
        self.headToTail = defaultdict(set)
        self.tailToHead = defaultdict(int)  # map from child to its own parent
        if track_restricted:
            self.buildRestrictedSymbolMap([])  # just initialize the attributes
        self.generating_children = defaultdict(set)  # mapping from symbol index to set of symbol indices it generates

    def compare(self, other, symbol_map=None):
        """Determines whether this AMR graph is the same as the AMR graph
        `other`. If symbol_map is None, this function assumes that the symbol
        sequences will be identical in the two graphs.

        Args:
            other: ULFAMRGraph to compare against
            symbol_map: dict from symbol ids of this graph to symbol ids in other

        Returns:
            True if the two graphs are the same, False otherwise.
        """
        assert isinstance(other, self.__class__), \
                "Comparing a graph to another class: %s" % other.__class__
        if self.n != other.n:
            # Not the same number of symbols.
            return False
        if symbol_map is not None and self.n != len(symbol_map) - 1:
            # Symbol map is incomplete.
            return False
        for i in range(self.n):
            first_symbol = self.symbols[i]
            j = symbol_map[i] if symbol_map is not None else i
            second_symbol = other.symbols[j]
            first_repr = first_symbol.symbol_repr(self)
            second_repr = second_symbol.symbol_repr(other)
            if first_repr != second_repr:
                print("Inconsistent symbol at %d: %s vs %s" % (i, first_repr, second_repr))

                return False
        return True

    def incomingArcs(self, v):
        if v < 0:
            return None
        symbol = self.symbols[v]
        return symbol.parent_rels

    def outgoingArcs(self, v):
        if v < 0 or v >= len(self.symbols):
            return []
        symbol = self.symbols[v]
        return symbol.rels

    def setRoot(self, v):
        self.root = v

    def count(self, restricted=False):
        self.counter += 1
        if not restricted:
            self.u_counter += 1

    def isAligned(self, v):
        if v >= len(self.symbols):
            return False
        return self.symbols[v].aligned

    def initTokens(self, toks):
        self.toks = toks

    def initLemma(self, lems):
        self.lemmas = lems

    def nextSymbolIDX(self, restricted_filtered=False):
        """Returns the next symbol index, optionally filtering out restricted symbols.
        """
        if restricted_filtered:
            return self.u_counter
        else:
            return self.counter

    def symbolLabel(self, idx):
        if idx < 0:
            return NULL
        symbol = self.symbols[idx]
        return symbol.getValue()

    def predSymbolLabel(self, idx):
        if idx < 0:
            return NULL
        symbol = self.pred_symbols[idx]
        return symbol.getValue()

    def getSymbol(self, idx):
        #print("getSymbol")
        #print("idx: {}".format(idx))
        #print("symbols: {}".format(self.symbols))
        if idx >= len(self.symbols):
            return None
        return self.symbols[idx]

    def getSymbolArc(self, symbol_idx, rel_idx, outgoing=True):
        if symbol_idx == -1:
            return NULL
        symbol = self.getSymbol(symbol_idx)
        if symbol is None:
            return NULL
        if outgoing:
            return symbol.getArc(rel_idx)
        return symbol.getParentArc(rel_idx)

    def getSymbolArcNum(self, symbol_idx, outgoing=True):
        if symbol_idx == -1:
            return 0
        symbol = self.getSymbol(symbol_idx)
        if symbol is None:
            return 0
        if outgoing:
            return len(symbol.rels)
        return len(symbol.parent_rels)

    def getSymbolSeq(self):
        return [symbol.getValue() for symbol in self.symbols]

    def getCategorySeq(self):
        return [symbol.category for symbol in self.symbols]

    def getMapInfoSeq(self):
        return [symbol.map_info for symbol in self.symbols]

    def getRestrictedSymbolSeq(self):
        """Returns the symbol value sequence for restricted symbols.
        """
        return [s.getValue() for s in self.r_symbols]

    def getRawSymbolSeq(self):
        """Returns the symbol value sequence for unrestricted symbols.
        """
        return [s.getValue() for s in self.u_symbols]

    def buildWordToSymbolIDX(self):
        """
        Assume multiple-to-one alignments
        :return:
        """
        for (i, symbol) in enumerate(self.symbols):
            for word_idx in symbol.alignments:
                self.widToSymbolID[word_idx] = i
                self.symIDToWordID[i] = word_idx
            span = symbol.span
            if span is not None:
                self.sidToSpan[i] = span

    def __str__(self):
        ret = ""
        for (i, symbol) in enumerate(self.symbols):
            ret += ("Current symbol %d: %s\n" % (i, symbol.getValue()))
            symbol.buildRelMap()
            rel_repr = ""
            for tail_v in symbol.tail_ids:
                rel_repr += symbol.rel_map[tail_v]
                rel_repr += (":" + self.symbols[tail_v].getValue() + "({})".format(tail_v) + " ")
            ret += ("Tail symbols: %s\n" % rel_repr)
        return ret

    def buildEdgeMap(self):
        # map an index to a list(right_edges list)
        right_edges_list = defaultdict(list)
        # for each symbol index and the symbol itself that is included in the array
        for (i, symbol) in enumerate(self.symbols):
            # check on what symbol does
            for tail_v in symbol.tail_ids:
                self.headToTail[i].add(tail_v)
                self.tailToHead[tail_v] = i
                if tail_v > i:
                    right_edges_list[i].append(tail_v)
                elif tail_v < i:
                    right_edges_list[tail_v].append(i)

        for left_idx in right_edges_list:
            sorted_right_list = sorted(right_edges_list[left_idx])
            assert sorted_right_list[0] > left_idx
            self.right_edges[left_idx] = sorted_right_list

    def addSymbol(self, symbol, restricted=False):
        """Adds the given symbol to the ULFAMRGraph with whether the symbol is a
        restricted symbol. The function returns three integers, (full symbol
        index, unrestricted symbol index, restricted symbol index), with a
        value of -1 if the given indexing is not available or applicable.
        """
        self.symbols.append(symbol)
        self.n += 1
        fidx = len(self.symbols) - 1
        # If we have restricted/unrestricted distinction, update that.
        if hasattr(self, "u_symbols") and self.u_symbols is not None:
            if restricted:
                self.r_symbols.append(symbol)
                ridx = len(self.r_symbols) - 1
                uidx = -1
                self.rtof[ridx] = fidx
                self.ftor[fidx] = ridx
            else:
                self.u_symbols.append(symbol)
                uidx = len(self.u_symbols) - 1
                ridx = -1
                self.utof[uidx] = fidx
                self.ftou[fidx] = uidx
        else:
            uidx = -1
            ridx = -1
        return fidx, uidx, fidx


    def buildRestrictedSymbolMap(self, restricted_symbols,
            inseq_symbols=None, promote_symbols=None):
        """Builds separate symbol indices for restricted symbols and other
        symbols, and a mapping between them.

        If inseq_symbols and promote_symbols are provided, these are labeled,
        under the assumption that these are subsets of restricted_symbols.
        """
        inseq_set = inseq_symbols if inseq_symbols is not None else []
        promote_set = promote_symbols if promote_symbols is not None else []
        # Restricted symbol information.
        r_symbol_info = [
                (i, s, s.getValue() in inseq_set, s.getValue() in promote_set)
                for i, s
                in enumerate(self.symbols)
                if s.getValue() in restricted_symbols
        ]
        # Restricted symbols.
        self.r_symbols = [s for _, s, _, _ in r_symbol_info]
        # Restricted symbols to full symbols mapping.
        self.rtof = { i: j for i, j in enumerate([i for i, _, _, _ in r_symbol_info]) }
        self.ftor = { j: i for i, j in self.rtof.items() }
        self.inseq_indices = set(i for i, _, is_inseq, _ in r_symbol_info if is_inseq)
        self.promote_indices = set(i for i, _, _, is_promote in r_symbol_info if is_promote)

        # Unrestricted symbol information.
        u_symbol_info = [
                (i, s)
                for i, s
                in enumerate(self.symbols)
                if s.getValue() not in restricted_symbols
        ]
        # Unrestricted symbols.
        self.u_symbols = [s for _, s in u_symbol_info]
        # Unrestricted symbols to full symbols mapping.
        self.utof = { i: j for i, j in enumerate([i for i, _ in u_symbol_info]) }
        self.ftou = { j: i for i, j in self.utof.items() }

    def isRestrictedIdx(self, symbol_idx):
        """Returns whether the given symbol index is a restricted index.
        """
        if not hasattr(self, "ftor"):
            return False
        return symbol_idx in self.ftor

    def isInSeqIdx(self, symbol_idx):
        if not hasattr(self, 'ftor'):
            return False
        return symbol_idx in self.inseq_indices

    def isPromoteIdx(self, symbol_idx):
        if not hasattr(self, 'ftor'):
            return False
        return symbol_idx in self.promote_indices

    def isAllReservedConstituent(self, symbol_idx):
        """Returns whether the given symbol index corresponds to a constituent
        only consisting of reserved symbols.
        """
        queue = [symbol_idx]
        traversed = []
        while len(queue) > 0:
            cur_idx = queue.pop()
            if not self.isRestrictedIdx(cur_idx):
                return False
            traversed.append(cur_idx)
            children = [i for i in self.headToTail[cur_idx] if i not in traversed]
            queue.extend(children)
        return True

    def uppermostInSeqIdx(self, symbol_idx):
        """Returns the uppermost symbol for the constituent rooted in the given
        symbol_idx that is an inseq symbol.
        """
        queue = deque([symbol_idx])
        traversed = []
        while len(queue) > 0:
            cur_idx = queue.popleft()
            if self.isInSeqIdx(cur_idx):
                return cur_idx
            traversed.append(cur_idx)
            children = [i for i in self.headToTail[cur_idx] if i not in traversed]
            queue.extend(children)
        return None

    def fillGeneratingChildren(self):
        """Fills in the map from symbol ids to a set of children it generates.
        This must be called after buildRestrictedSymbolMap and buildEdgeMap.
        """
        # First go through graph from leaf nodes, going upward, which nodes
        # have a non-restricted node as a descendent.
        # Then, any restricted node without a non-restricted descendent can
        # be generated by its parent.
        leaf_node_idxs = [
                i for i, s in enumerate(self.symbols)
                if len(s.tail_ids) == 0
        ]
        # Symbol idx to whether it has a non-restricted desc.
        idx2pred = {}
        queue = deque(leaf_node_idxs)
        accounted = leaf_node_idxs
        while len(queue) > 0:
            cur_idx = queue.popleft()
            cur_symbol = self.symbols[cur_idx]
            has_desc = False
            for child_idx in cur_symbol.tail_ids:
                if not self.isRestrictedIdx(child_idx) or idx2pred[child_idx]:
                    has_desc = True
                    break
            idx2pred[cur_idx] = has_desc
            for parent_idx in cur_symbol.parent_ids:
                # Add if all the children have already been added.
                if (parent_idx not in accounted and
                        all([child in accounted
                            for child
                            in self.symbols[parent_idx].tail_ids])):
                    queue.append(parent_idx)
                    accounted.append(parent_idx)

        assert len(accounted) == len(self.symbols)

        # Any symbol that does not have a non-restricted descendent and
        # is restricted itself is generated by its parent.
        for idx in range(len(self.symbols)):
            if not idx2pred[idx] and self.isRestrictedIdx(idx):
                cur_symbol = self.symbols[idx]
                for parent_idx in cur_symbol.parent_ids:
                    self.generating_children[parent_idx].add(idx)

    def buildUnrestrictedDescendantMap(self):
        """This must be called after buildRestrictedSymbolMap.
        It fills in an internal dictionary u_descendents, which contains the
        unrestricted descendant for each node in the graph. This assumes that
        the graph is in fact more restricted as a tree.
        """
        # Perform reverse DFS to get all children before parents.
        # First build the stack of indices.
        stack = [self.root]
        traverse_stack = [self.root]
        while len(traverse_stack) > 0:
            cur_idx = traverse_stack.pop()
            cur_indices = [x for x in list(self.headToTail[cur_idx]) if x not in stack]
            traverse_stack.extend(cur_indices)
            stack.extend(cur_indices)

        # Traverse and build descendants.
        u_desc_list = defaultdict(list)
        while len(stack) > 0:
            cur_idx = stack.pop()
            children = list(self.headToTail[cur_idx])
            u_children = [x for x in children if x in self.ftou]
            child_u_descs = [
                    d for descs in [u_desc_list[sidx] for sidx in children]
                    for d in descs
            ]
            cur_u_descs = list(set(u_children + child_u_descs))
            u_desc_list[cur_idx] = sorted(cur_u_descs)

        self.u_desc_fidx = u_desc_list
        self.u_desc_uidx = {
                self.ftou[k]: [self.ftou[v] for v in vs]
                for k, vs in u_desc_list.items()
                if k in self.ftou
        }


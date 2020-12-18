import sys
import os
from collections import deque, namedtuple
from .dependency import *
from .ULFAMRGraph import *
# from postprocessing import AMR_seq
import copy
from .focus import NextSymbolData
from .utils import (
    ArcDir, FeatureType, TokenType, UnreserveGenMethod,
    END_WIDX, NONE_WIDX, NULL_WIDX, NULL_SIDX, NULL,
    cache_feat_num, shiftpop_feat_num, pushidx_feat_num,
    symselect_feat_num, 
)
# for type checking
from lisp.composition import MemoizedULFTypes
from .exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()

TransitionSystemVertex = namedtuple("TransitionSystemVertex", ["word_idx", "symbol_idx"])

# Mapping for special tokens that don't behave how we might expect.
SPECIAL_TOKEN_MAP = {
    "''": "\"|\\\"|\"",
    "'": "\"|\\\"|\"",
    "``": "\"|\\\"|\"",
    "`": "\"|\\\"|\"",
}

class CacheConfiguration(object):
    """Class holding the cache transition system state.
    """
    def __init__(self, size, length, config=None, oracle=False):
        """
        :param size: number of elems of the fixed-sized cache.
        :param length: number of words in the buffer.
        """
        self.cache_size = size
        if config is None:
            self.stack = []
            self.widx_buffer = deque(range(length))
            self.widx_width = 1

            # Each cache elem is a (word idx, symbol idx) pair.
            self.cache = [TransitionSystemVertex(NULL_WIDX, NULL_SIDX) for _ in range(size)]

            self.candidate = None   # A register for the newly generated vertex.

            self.hypothesis = ULFAMRGraph(track_restricted=True)  # The ULFAMR graph being built.
            self.gold = None  # The reference ULFAMR graph.

            self.wordSeq, self.lemSeq, self.posSeq, self.nerSeq = [], [], [], []
            # Sentence annotation info (input to decoding)
            self.symbolSeq, self.symbolAlign, self.actionSeq = [], [], []
            # Mapping to-from self.symbolSeq and self.hypothesis.symbols
            self.next_symbol_data = NextSymbolData()
            self.next_word_idx = 0

            self.start_word = True # Whether start processing a new word.
            self.tree = DependencyTree()
            self.cand_vertex = None  # Candidate vertex
            self.last_action = None
            self.pop_widx_buff = True
            self.phase = FeatureType.SHIFTPOP
            self.widTosid = {}  # word idx to symbol idx

            # dictionary so that one can get the types all at once without consulting cl4py each time
            # as that takes time
            self.ulf_type_fn = MemoizedULFTypes().get_type
            # Mapping from the hypothesis symbol index to ULF type. This takes into account compositions,
            # so it will vary based on arcs between symbols.
            self.sidx2type = []
            # Composition type determined during type checking. Store it to
            # update Configuration if action is actually taken.
            self.next_compose_type = None
            if oracle:
                self.symbol_available = None
            else:
                self.symbol_available = []

            # symbol correspondences between gold and hyp graphs.
            self.hyp2gold = {-1: -1}
            self.gold2hyp = {-1: -1}

        else:
            self.stack = copy.copy(config.stack)
            self.widx_buffer = copy.copy(config.widx_buffer)
            self.widx_width = copy.copy(config.widx_width)
            self.cache = copy.copy(config.cache)
            self.wordSeq, self.lemSeq, self.posSeq, self.nerSeq = config.wordSeq, config.lemSeq, config.posSeq, config.nerSeq
            self.symbolSeq = config.symbolSeq
            self.actionSeq = config.actionSeq
            self.next_symbol_data = copy.deepcopy(config.next_symbol_data)
            self.next_word_idx = config.next_word_idx
            self.tree, self.symbolAlign = config.tree, config.symbolAlign
            self.phase = config.phase
            self.hypothesis = copy.deepcopy(config.hypothesis)
            self.start_word = config.start_word
            self.cand_vertex = copy.copy(config.cand_vertex)
            self.pop_widx_buff = config.pop_widx_buff
            self.last_action = config.last_action
            self.widTosid = copy.copy(config.widTosid)
            self.ulf_type_fn = config.ulf_type_fn
            self.sidx2type = copy.deepcopy(config.sidx2type)
            self.next_compose_type = config.next_compose_type
            self.hyp2gold = copy.copy(config.hyp2gold)
            self.gold2hyp = copy.copy(config.gold2hyp)
            self.symbol_available = copy.deepcopy(config.symbol_available)

    def setGold(self, graph_, update_widx=True):
        """Sets the gold graph for oracle decoding.
        """
        self.gold = graph_
        if update_widx:
            widx_buf = deque()
            for i in range(len(self.gold.symbols)):
                if self.gold.isAligned(i):
                    widx_buf.append(self.gold.symIDToWordID[i])
            self.widx_buffer = widx_buf

    def buildWordToSymbol(self):
        assert (len(self.symbolAlign) == 0 and len(self.widTosid) == 0) or \
                len(self.symbolAlign) > 0
        for (sidx, aligned_widx) in enumerate(self.symbolAlign):
            if aligned_widx != NONE_WIDX:
                self.widTosid[aligned_widx] = sidx

    # For type system checking
    # as unknown type will be ..., just uses that as the type temporarily
    def buildSymbolToTypes(self):
        sentence = ""
        #symbol_seq = self.symbolSeq
        symbol_seq = [s.getValue() for s in self.hypothesis.symbols]
        # Fill the memo with the symbol-type correspondence.
        for i in range(len(symbol_seq)):
            symbol = self.symbol_seq[i]
            label = '(' + symbol + ')'
            #print("symbol: {}".format(symbol))
            #print(os.getcwd())
            self.sidx2type.append(self.ulf_type_fn(symbol))
            sentence += "Index: %d, Symbol: %s, Type: %s \n" %(i, symbol, self.sidx2type[i])
        #print(sentence)

    def addAtomicULFType(self, sidx):
        """Adds the ULF type for the given symbol index. If this mapping hasn't been computed yet,
        add it and mark all symbols between the last typed and current one as SKIPPED. If SKIPPED,
        update it with a computed value.

        This doesn't need to be updated since it computes atomic types. For composed types, refer
        to the connectArc function.
        """
        if len(self.sidx2type) >= sidx:
            for new_sidx in range(len(self.sidx2type), sidx):
                self.sidx2type.append("SKIPPED")
            new_type = self.ulf_type_fn(self.hypothesis.symbols[sidx].getValue())
            self.sidx2type.append(new_type)
            #print("Added new type: {}\nAt symbol index:{}For symbol: {}\n".format(new_type, sidx, self.hypothesis.symbols[sidx]))
        elif self.sidx2type[sidx] == "SKIPPED":
            new_type = self.ulf_type_fn(self.hypothesis.symbols[sidx].getValue())
            self.sidx2type[sidx] = new_type
            #print("Added new type: {}\nAt symbol index:{}".format(new_type, sidx))
        else:
            raise Exception("No type added to sidx: {}".format(sidx))

    def buildSymbolAvailability(self):
        for i in range(len(self.symbolSeq)):
            self.symbol_available[i] = True

    def isUnalign(self, idx):
        return self.symbolAlign[idx] == NONE_WIDX

    def getSymbol(self, idx, orig=True):
        """Gets the symbol with the corresponding index.
        orig flag indicates whether to use the original symbol sequence
        indexing or the post-generated symbol sequence indexing.

        If orig == False and idx is greater than the last generated symbol,
        assume continuation of mapping from all to orig through the mapping.

        Returns None if the index is out-of-range.
        """
        if orig and idx >= 0  and idx < len(self.symbolSeq):
            return self.symbolSeq[idx]
        elif orig:
            return None
        elif idx < len(self.hypothesis.symbols):
            return self.hypothesis.symbols[idx].getValue()
        elif len(self.next_symbol_data.all2orig) == 0:
            return None
            #return self.symbolSeq[0] if len(self.symbolSeq) > 0 else None
        else:
            max_all = max(self.next_symbol_data.all2orig.keys())
            last_idx = len(self.hypothesis.symbols) - 1
            orig_idx = (
                    self.next_symbol_data.all2orig[idx]
                    if idx in self.next_symbol_data.all2orig
                    else self.next_symbol_data.all2orig[max_all] + max((idx - last_idx), 1)
            )
            if orig_idx >= len(self.symbolSeq):
                return None
            else:
                return self.symbolSeq[orig_idx]

    # Get the token feature
    # A description string for the token
    def getTokenFeats(self, idx, type):
        if idx < 0:
            return type.name + ":" + NULL
        if type != TokenType.SYMBOL and idx >= len(self.wordSeq):
            return type.name + ":" + NULL
        prefix = type.name + ":"
        if type == TokenType.WORD:
            return prefix + self.wordSeq[idx]
        if type == TokenType.LEM:
            return prefix + self.lemSeq[idx]
        if type == TokenType.POS:
            return prefix + self.posSeq[idx]
        if type == TokenType.NER:
            return prefix + self.nerSeq[idx]
        if type == TokenType.SYMBOL:
            symbol = self.getSymbol(idx, orig=False)
            if not symbol:
                return type.name + ":" + NULL
            return prefix + symbol

    def getArcFeats(self, symbol_idx, idx, prefix, outgoing=True):
        arc_label = self.hypothesis.getSymbolArc(symbol_idx, idx, outgoing)
        return prefix + arc_label

    def getNumArcFeats(self, symbol_idx, prefix, outgoing=True):
        arc_num = self.hypothesis.getSymbolArcNum(symbol_idx, outgoing)
        return "%s%d" % (prefix, arc_num)

    def getDepDistFeats(self, idx1, idx2):
        prefix = "DepDist="
        if idx1 < 0 or idx2 < 0:
            return prefix + NULL
        dep_dist = self.tree.getDepDist(idx1, idx2)
        return "%s%d" % (prefix, dep_dist)

    def getDepLabelFeats(self, idx, feats, prefix="dep", k=3):
        dep_arcs = self.tree.getAllChildren(idx)
        for i in range(k):
            curr_dep = prefix + ":" + NULL
            if i < len(dep_arcs):
                curr_dep = prefix + ":" + dep_arcs[i]
            feats.append(curr_dep)

    def getDepParentFeat(self, idx, feats, prefix="pdep"):
        parent_arc = self.tree.getLabel(idx)
        curr_dep = prefix + ":" + NULL
        if parent_arc:
            curr_dep = prefix + ":" + parent_arc
        feats.append(curr_dep)

    def getTokenDistFeats(self, idx1, idx2, upper, prefix):
        if idx1 < 0 or idx2 < 0:
            return prefix + NULL
        #assert idx1 < idx2, "Left token index not smaller than right"
        if idx1 < idx2:
            token_dist = idx2 - idx1
        else:
            token_dist = idx1 - idx2
        if token_dist > upper:
            token_dist = upper
        return "%s%d" % (prefix, token_dist)

    def getTokenTypeFeatures(self, word_idx, symbol_idx, feats, prefix=""):
        word_repr = prefix + self.getTokenFeats(word_idx, TokenType.WORD)
        lem_repr = prefix + self.getTokenFeats(word_idx, TokenType.LEM)
        pos_repr = prefix + self.getTokenFeats(word_idx, TokenType.POS)
        ner_repr = prefix + self.getTokenFeats(word_idx, TokenType.NER)
        symbol_repr = prefix + self.getTokenFeats(symbol_idx, TokenType.SYMBOL)
        feats.append(word_repr)
        feats.append(lem_repr)
        feats.append(pos_repr)
        feats.append(ner_repr)
        feats.append(symbol_repr)

    def getSymbolRelationFeatures(self, symbol_idx, feats):
        first_symbol_arc = self.getArcFeats(symbol_idx, 0, "ARC=")
        second_symbol_arc = self.getArcFeats(symbol_idx, 1, "ARC=")
        parent_symbol_arc = self.getArcFeats(symbol_idx, 0, "PARC=", False)
        symbol_parrel_num = self.getNumArcFeats(symbol_idx, "NPARC=", False)
        feats.append(first_symbol_arc)
        feats.append(second_symbol_arc)
        feats.append(parent_symbol_arc)
        feats.append(symbol_parrel_num)

    def getCacheFeat(self, word_idx=-1, symbol_idx=-1, idx=-1, uniform_arc=True, arc_label=False):
        if idx == -1:
            if uniform_arc or arc_label:
                return ["NONE"] * (cache_feat_num() + 8)
            return ["NONE"] * cache_feat_num()

        feats = []
        cache_word_idx, cache_symbol_idx = self.getCache(idx)

        # Candidate token features.
        self.getTokenTypeFeatures(word_idx, symbol_idx, feats)

        # Cache token features.
        self.getTokenTypeFeatures(cache_word_idx, cache_symbol_idx, feats)

        # Distance features
        word_dist_repr = self.getTokenDistFeats(cache_word_idx, word_idx, 4, "WordDist=")
        symbol_dist_repr = self.getTokenDistFeats(cache_symbol_idx, symbol_idx, 4, "SymbolDist=")
        dep_dist_repr = self.getDepDistFeats(cache_word_idx, word_idx)

        # Dependency label
        dep_label_repr = "DepLabel=" + self.tree.getDepLabel(cache_word_idx, word_idx)

        feats.append(word_dist_repr)
        feats.append(symbol_dist_repr)
        feats.append(dep_dist_repr)
        feats.append(dep_label_repr)

        # If the arc label, then extract all the dependency label features.
        if arc_label:
            self.getDepLabelFeats(word_idx, feats, "dep", 3)
            self.getDepParentFeat(word_idx, feats, "pdep")
            self.getDepLabelFeats(cache_word_idx, feats, "cdep", 3)
            self.getDepParentFeat(cache_word_idx, feats, "cpdep")
        elif uniform_arc:
            feats += ["NONE"] * 8

        # Get arc information for the current symbol
        self.getSymbolRelationFeatures(symbol_idx, feats)

        self.getSymbolRelationFeatures(cache_symbol_idx, feats)

        if arc_label or uniform_arc:
            assert len(feats) == cache_feat_num() + 8, \
                    "len(feats): {}\ncache_feat_num: {}\nfeats: {}\n".format(
                            len(feats), cache_feat_num(), feats)
        else:
            assert len(feats) == cache_feat_num(), \
                    "len(feats): {}\ncache_feat_num: {}\nfeats: {}\n".format(
                            len(feats), cache_feat_num(), feats)
        return feats

    def pushIDXFeatures(self, word_idx=-1, symbol_idx=-1):
        """Features for PUSHIDX. Token type features for the candidate vertex
        and all cache vertices.
        """
        if symbol_idx == -1:
            return ["NONE"] * pushidx_feat_num(self.cache_size)
        feats = []

        # Candidate vertex features.
        self.getTokenTypeFeatures(word_idx, symbol_idx, feats)

        # Cache vertex features.
        for cache_idx in range(self.cache_size):
            cache_word_idx, cache_symbol_idx = self.getCache(cache_idx)
            prefix = "cache%d_" % cache_idx
            self.getTokenTypeFeatures(cache_word_idx, cache_symbol_idx, feats, prefix)

        assert len(feats) == pushidx_feat_num(self.cache_size), \
                "len(feats): {}\npushidx_feat_num: {}\nfeats: {}\n".format(
                        len(feats), pushidx_feat_num(self.cache_size), feats)
        return feats

    def shiftPopFeatures(self, word_idx=-1, symbol_idx=-1, active=False):
        """Features for SHIFTPOP. Token type features for the rightmost cache
        value, the next word/symbol token type features, and the dependency
        links from word associated with the rightmost cache value.
        """
        if not active:
            return ["NONE"] * shiftpop_feat_num()
        feats = []
        rst_word_idx, rst_symbol_idx = self.rightmostCache()
        # Right most cache token features
        self.getTokenTypeFeatures(rst_word_idx, rst_symbol_idx, feats, "rst_")

        # Word index buffer token features
        self.getTokenTypeFeatures(word_idx, symbol_idx, feats, "buf_")

        # Then get the dependency links to right words
        dep_list = self.wordDepConnections(rst_word_idx)

        dep_num = len(dep_list)
        if dep_num > 4:
            dep_num = 4
        dep_num_repr = "depnum=%d" % dep_num
        feats.append(dep_num_repr)
        for i in range(3):
            if i >= dep_num:
                feats.append("dep=" + NULL)
            else:
                feats.append("dep=" + dep_list[i])

        # Outgoing ULF arcs (up to 4) of the rightmost cache value.
        arc_list = self.hypothesis.outgoingArcs(rst_symbol_idx)
        arc_num = len(arc_list)
        if arc_num > 4:
            arc_num = 4
        arc_num_repr = "rst_arcnum=%d" % arc_num
        feats.append(arc_num_repr)
        for i in range(3):
            if i >= arc_num:
                feats.append("rst_arc=" + NULL)
            else:
                feats.append("rst_arc=" + arc_list[i])

        assert len(feats) == shiftpop_feat_num(), \
                "len(feats): {}\nshiftpop_feat_num: {}\nfeats: {}\n".format(
                        len(feats), shiftpop_feat_num(), feats)
        return feats

    def symSelectFeatures(self, word_idx=-1, symbol_idx=-1, active=False):
        """Features for SYMSELECT.
        - token features for the rightmost cache element
        - token features for 3 buffer elements centered on current one
        """
        if not active:
            return ["NONE"] * symselect_feat_num()
        feats = []
        rst_widx, rst_sidx = self.rightmostCache()
        # Rightmost cache token features.
        self.getTokenTypeFeatures(rst_widx, rst_sidx, feats, "rst_")
        # Word index buffer token features.
        #for i in range(-1, 2):
        #    self.getTokenTypeFeatures(word_idx, symbol_idx + i, feats, "buf{}_".format(i))
        for i in range(0, 1):
            self.getTokenTypeFeatures(word_idx, symbol_idx + i, feats, "buf{}_".format(i))
        assert len(feats) == symselect_feat_num(), \
                "len(feats): {}\nsymselect_feat_num: {}\n feats: {}\n".format(
                        len(feats), symselect_feat_num(), feats)
        return feats

    def extractFeatures(self, feature_type, word_idx=-1, symbol_idx=-1, cache_idx=-1, uniform_arc=False):
        """Generates features for the cache transition system. Every feature
        index is always used, but features not associated with the given feature
        type will be set to NONE.

        Assumes that the symbol_idx refers to index of the original symbolSeq
        without generated symbols.
        """
        phase_feat = "PHASE="

        # SHIFTPOP
        if feature_type == FeatureType.SHIFTPOP:
            shiftpop_feats = self.shiftPopFeatures(word_idx, symbol_idx, True)
            phase_feat += "SHIFTPOP"
        else:
            shiftpop_feats = self.shiftPopFeatures()

        # ARCBINARY/ARCCONNECT
        if feature_type == FeatureType.ARCBINARY:
            phase_feat += "ARCBINARY"
            if uniform_arc: # cache_feats + 8
                cache_feats = self.getCacheFeat(word_idx, symbol_idx, cache_idx,
                                                uniform_arc=True, arc_label=False)
            else:
                binary_feats = self.getCacheFeat(word_idx, symbol_idx, cache_idx,
                                                 uniform_arc=False, arc_label=False)
                label_feats = self.getCacheFeat(uniform_arc=False, arc_label=True)
                cache_feats = binary_feats + label_feats
        elif feature_type == FeatureType.ARCCONNECT:
            phase_feat += "ARCLABEL"
            if uniform_arc: # cache_feats + 8
                cache_feats = self.getCacheFeat(word_idx, symbol_idx, cache_idx,
                                                uniform_arc=True, arc_label=True)
            else:
                binary_feats = self.getCacheFeat(uniform_arc=False, arc_label=False)
                label_feats = self.getCacheFeat(word_idx, symbol_idx, cache_idx,
                                                uniform_arc=False, arc_label=True)
                cache_feats = binary_feats + label_feats
        else:
            if uniform_arc:
                cache_feats = self.getCacheFeat(uniform_arc=True)
            else:
                binary_feats = self.getCacheFeat(uniform_arc=False, arc_label=False)
                label_feats = self.getCacheFeat(uniform_arc=False, arc_label=True)
                cache_feats = binary_feats + label_feats

        # PUSHIDX
        if feature_type == FeatureType.PUSHIDX:
            phase_feat += "PUSHIDX"
            pushidx_feats = self.pushIDXFeatures(word_idx, symbol_idx)
        else:
            pushidx_feats = self.pushIDXFeatures()

        # SYMSELECT
        if feature_type == FeatureType.SYMSELECT:
            phase_feat += "SYMSELECT"
            symsel_feats = self.symSelectFeatures(word_idx, symbol_idx, True)
        else:
            symsel_feats = self.symSelectFeatures()

        # WORDGEN
        if feature_type == FeatureType.WORDGEN:
            phase_feat += "WORDGEN"
            # Just use the shiftpop_feats for now.
            shiftpop_feats = self.shiftPopFeatures(word_idx, symbol_idx, True)
        # WORDGEN_*
        # Just use the shiftpop_feats for now.
        if feature_type == FeatureType.WORDGEN_LEMMA:
            phase_feat += "WORDGEN_LEMMA"
            shiftpop_feats = self.shiftPopFeatures(word_idx, symbol_idx, True)
        if feature_type == FeatureType.WORDGEN_NAME:
            phase_feat += "WORDGEN_NAME"
            shiftpop_feats = self.shiftPopFeatures(word_idx, symbol_idx, True)
        if feature_type == FeatureType.WORDGEN_TOKEN:
            phase_feat += "WORDGEN_TOKEN"
            shiftpop_feats = self.shiftPopFeatures(word_idx, symbol_idx, True)

        # PROMOTE*
        if feature_type == FeatureType.PROMOTE:
            phase_feat += "PROMOTE"
            #promote_feats = self.shiftPopFeatures(word_idx, symbol_idx, True)
            binary_feats = self.getCacheFeat(uniform_arc=False, arc_label=False)
            label_feats = self.getCacheFeat(word_idx, symbol_idx, cache_idx,
                                            uniform_arc=False, arc_label=True)
            promote_feats = binary_feats + label_feats
        elif feature_type == FeatureType.PROMOTE_SYM:
            phase_feat += "PROMOTE_SYM"
            #promote_feats = self.shiftPopFeatures(word_idx, symbol_idx, True)
            binary_feats = self.getCacheFeat(uniform_arc=False, arc_label=False)
            label_feats = self.getCacheFeat(word_idx, symbol_idx, cache_idx,
                                            uniform_arc=False, arc_label=True)
            promote_feats = binary_feats + label_feats
        elif feature_type == FeatureType.PROMOTE_ARC:
            phase_feat += "PROMOTE_ARC"
            #promote_feats = self.shiftPopFeatures(word_idx, symbol_idx, True)
            binary_feats = self.getCacheFeat(uniform_arc=False, arc_label=False)
            label_feats = self.getCacheFeat(word_idx, symbol_idx, cache_idx,
                                            uniform_arc=False, arc_label=True)
            promote_feats = binary_feats + label_feats
        else:
            #promote_feats = self.shiftPopFeatures()
            binary_feats = self.getCacheFeat(uniform_arc=False, arc_label=False)
            label_feats = self.getCacheFeat(uniform_arc=False, arc_label=True)
            promote_feats = binary_feats + label_feats

        assert phase_feat != "PHASE="
        retval = [phase_feat] + shiftpop_feats + cache_feats + pushidx_feats + symsel_feats + promote_feats
        #print("extracted feature dimentions: {}".format(len(retval)))
        #print("shiftpop dim: {}".format(len(shiftpop_feats)))
        #print("cache_feats dim: {}".format(len(cache_feats)))
        #print("pushidx_feats dim: {}".format(len(pushidx_feats)))
        #print("symsel feat dim: {}".format(len(symsel_feats)))
        return retval

    def clearWidxBuffer(self):
        self.widx_buffer.clear()

    def popWidxBuffer(self):
        pop_elem = self.widx_buffer.popleft()
        return pop_elem

    def peekWordIdx(self, idx=0):
        if len(self.widx_buffer) <= idx:
            return NONE_WIDX
        # assert len(self.widx_buffer) > 0, "Fetch word from empty word index buffer."
        return self.widx_buffer[idx]

    def getCurToken(self, delim=" ", idx_offset=0, width_override=None):
        if len(self.widx_buffer) == 0:
            return ""
        else:
            widx = self.peekWordIdx() + idx_offset
            width = self.widx_width if width_override is None else width_override
            token = delim.join(self.wordSeq[widx:widx + width])
            if token in SPECIAL_TOKEN_MAP:
                token = SPECIAL_TOKEN_MAP[token]
            return token

    def getCurLemma(self, delim="_", idx_offset=0, width_override=None):
        if len(self.widx_buffer) == 0:
            return ""
        else:
            widx = self.peekWordIdx() + idx_offset
            width = self.widx_width if width_override is None else width_override
            lemma = delim.join(self.lemSeq[widx:widx + width])
            if lemma in SPECIAL_TOKEN_MAP:
                lemma = SPECIAL_TOKEN_MAP[lemma]
            return lemma

    def getCurWord(self, delim=None, idx_offset=0, width_override=None, phase=None):
        if phase == None:
            phase = self.phase
        # Return nothing if not in the right phase.
        if phase not in [
            FeatureType.WORDGEN_TOKEN,
            FeatureType.WORDGEN_LEMMA,
            FeatureType.WORDGEN_NAME,
        ]:
            return ""
        if phase == FeatureType.WORDGEN_LEMMA:
            delim = "_" if delim is None else delim
            return self.getCurLemma(delim, idx_offset, width_override)
        else:
            delim = " " if delim is None else delim
            return self.getCurToken(delim, idx_offset, width_override)

    def getNextToken(self, width_override=None):
        if len(self.widx_buffer) <= self.widx_width:
            return ""
        else:
            width = self.widx_width if width_override is None else width_override
            token = self.wordSeq[self.peekWordIdx(width)]
            if token in SPECIAL_TOKEN_MAP:
                token = SPECIAL_TOKEN_MAP[token]
            return token

    def getNextLemma(self, width_override=None):
        if len(self.widx_buffer) <= self.widx_width:
            return ""
        else:
            width = self.widx_width if width_override is None else width_override
            lemma = self.lemSeq[self.peekWordIdx(width)]
            if lemma in SPECIAL_TOKEN_MAP:
                lemma = SPECIAL_TOKEN_MAP[lemma]
            return lemma

    def widxBufferSize(self):
        return len(self.widx_buffer)

    def getCache(self, idx):
        #print("idx: {}".format(idx))
        #print("type(idx): {}".format(type(idx)))
        if idx < 0 or idx >= self.cache_size:
            return (None, None)
        return self.cache[idx]

    def getCacheSymbol(self, cache_idx, type=TokenType.SYMBOL):
        _, symbol_idx = self.getCache(cache_idx)
        return self.hypothesis.symbols[symbol_idx].getValue()

    def getConnectedArcs(self, cache_idx, left=True):
        _, symbol_idx = self.getCache(cache_idx)
        outgoing_rels = self.hypothesis.outgoingArcs(symbol_idx)
        if left:
            return set(["R-%s" % l for l in outgoing_rels if (l != "op" and l != "mod")])
        return set(["L-%s" % l for l in outgoing_rels if (l != "op" and l != "mod")])

    def moveToCache(self, elem):
        self.cache.append(elem)

    def popStack(self):
        stack_size = len(self.stack)
        if stack_size < 1:
            return None
        top_elem = self.stack.pop()
        return top_elem

    def getStack(self, idx):
        stack_size = len(self.stack)
        if idx < 0 or idx >= stack_size:
            return None
        return self.stack[idx]

    def stackSize(self):
        return len(self.stack)

    # Whether the next operation should be processing a new word
    # or a vertex from the cache.
    def needsPop(self):
        """
        Whether the next operation should be processing a new word
        or a vertex from the cache.
        :return:
        """

        """
        # cannot just compare length, no longer true that if process all symbol
        # already one can pop
        last_cache_word, cache_hypo_symbol_idx = self.cache[self.cache_size-1]
        if cache_hypo_symbol_idx == -1:
            return False
        """
        last_cache_word, cache_hypo_symbol_idx = self.rightmostCache()
        # in the case when the cache does not store any useful information yet
        if cache_hypo_symbol_idx == NULL_SIDX:
            return False
        # print "Current last cache word %d, cache symbol %d" % (last_cache_word, cache_hypo_symbol_idx)
        # ($, $) at last cache position.
        # Can pop if the rightmost element has all of its edges.
        hypo_symbol = self.hypothesis.symbols[cache_hypo_symbol_idx]
        cache_gold_symbol_idx = self.hyp2gold[cache_hypo_symbol_idx]
        gold_symbol = self.gold.symbols[cache_gold_symbol_idx]
        return len(hypo_symbol.tail_ids) == len(gold_symbol.tail_ids) and len(hypo_symbol.parent_ids) == len(gold_symbol.parent_ids)

        #next_symbol_idx = self.hypothesis.nextSymbolIDX()
        #num_symbols = self.gold.n

        ## if already taken in all symbols
        #if next_symbol_idx >= num_symbols:
        #    print("RET COND: Taken all symbols in.")
        #    return True

        #right_edges = self.gold.right_edges
        ## if the last symbol does not have any right_edges, can pop and still connect
        #if cache_gold_symbol_idx not in right_edges:
        #    # as flow changes, as long as we have not yet seen all symbols, it is possible that
        #    # the child has been seen but not connected yet. Thus not pop in that situation
        #    # here i use symbol since this indicates an action has been selected and applied successfully
        #    if len(self.hypothesis.symbols[cache_hypo_symbol_idx].tail_ids) < len(self.gold.symbols[cache_gold_symbol_idx].tail_ids):
        #        return False
        #    return True

        #assert next_symbol_idx > cache_hypo_symbol_idx and num_symbols > cache_gold_symbol_idx

        ## -1 refers to the last element in the array
        ## if all right_edges are made, pop the element out
        #return right_edges[cache_gold_symbol_idx][-1] < next_symbol_idx

    def needsPromote(self):
        """
        Whether a promote is needed for the last symbol
        in the cache
        - Symbol label, hypo_graph index and gold_graph index
          if all children supplied and connected to 
          restricted parent
        - None otherwise
        """

        cache_word_idx, hypo_cache_symbol_idx = self.rightmostCache()
        if hypo_cache_symbol_idx == NULL_SIDX: # nil no promote
            return None, None, None

        gold_cache_symbol_idx = self.hyp2gold[hypo_cache_symbol_idx] #map to gold graph index

        hypo_symbol = self.hypothesis.symbols[hypo_cache_symbol_idx]
        gold_cache_symbol_idx = self.hyp2gold[hypo_cache_symbol_idx]
        gold_symbol = self.gold.symbols[gold_cache_symbol_idx]
        # if still have unconnected child, no promote
        if len(hypo_symbol.tail_ids) != len(gold_symbol.tail_ids):
            return None, None, None
        # obtain parent index
        parent_idx = self.gold.tailToHead[gold_cache_symbol_idx]
        # if not restricted symbol
        if not self.gold.isRestrictedIdx(parent_idx):
            return None, None, None
        # check if the symbol has been generated
        if parent_idx in self.gold2hyp:
            return None, None, None
        parent_label = self.gold.symbolLabel(parent_idx)
        hypo_idx = self.hypothesis.nextSymbolIDX()
        return parent_label, hypo_idx, parent_idx

    def shiftWidxBuffer(self):
        if len(self.widx_buffer) == 0:
            return False
        self.popWidxBuffer()
        return True

    def rightmostCache(self):
        return self.cache[self.cache_size - 1]

    def pop(self):
        """Performs a stack pop action. Pops the top element in the stack into
        the index it was original pushed from, and pushes out the rightmost
        element in the cache.

        Returns whether it was successful. This fails when the stack is empty.
        """
        stack_size = len(self.stack)
        if stack_size < 1:
            return False
        cache_idx, vertex = self.stack.pop()
        # Insert a vertex to a certain cache position.
        # Then pop the last cache vertex.
        self.cache.insert(cache_idx, vertex)
        self.cache.pop()
        self.cand_vertex = self.rightmostCache()
        return True

    def connectArc(self, cache_vertex_idx, cand_vertex_idx, direction, arc_label,
            use_types=False):
        """
        Make a directed labeled arc between a cache vertex and the candidate vertex.
        :param cache_vertex_idx: a symbol index for a vertex in the cache.
        :param cand_vertex_idx: a symbol index for a candidate symbol that was generated.
        :param direction: the direction of the connected arc.
        :param arc_label: the label of the arc.
        :param use_types: whether the configuration should update type information.
        :return: None
        """
        cand_s = self.hypothesis.symbols[cand_vertex_idx]
        cache_s = self.hypothesis.symbols[cache_vertex_idx]
        if direction == ArcDir.LEFT: # an L-edge, from candidate to cache.
            cand_s.tail_ids.append(cache_vertex_idx)
            cand_s.rels.append(arc_label)
            cache_s.parent_ids.append(cand_vertex_idx)
            cache_s.parent_rels.append(arc_label)
            # move the type to the parent
            if use_types:
                self.sidx2type[cand_vertex_idx] = self.next_compose_type
            # remove availability of symbol that was the child.
            if self.symbol_available is not None:
                self.symbol_available[cache_vertex_idx] = False
        else:
            cand_s.parent_ids.append(cache_vertex_idx)
            cand_s.parent_rels.append(arc_label)
            cache_s.tail_ids.append(cand_vertex_idx)
            cache_s.rels.append(arc_label)
            # same as above
            if use_types:
                self.sidx2type[cache_vertex_idx] = self.next_compose_type
            if self.symbol_available is not None:
                self.symbol_available[cand_vertex_idx] = False
    
    def wordDepConnections(self, word_idx, thre=20):
        ret_set = set()
        start = self.tree.n
        if len(self.widx_buffer) > 0:
            start = self.widx_buffer[0]
        end = self.tree.n
        if end - start > thre:
            end = start + thre
        for idx in range(start, end):
            if self.tree.getHead(idx) == word_idx:
                ret_set.add("R="+self.tree.getLabel(idx))
            elif self.tree.getHead(word_idx) == idx:
                ret_set.add("L="+self.tree.getLabel(word_idx))
        return list(ret_set)

    def wordDepConnection(self, word_idx, thre=20):
        ret_set = self.wordDepConnections(word_idx, thre)
        return len(ret_set)

    def pushToStack(self, cache_idx):
        """
        Push a certain cache vertex onto the stack.
        :param cache_idx:
        :return:
        """
        cache_word_idx, cache_symbol_idx = self.cache[cache_idx]
        del self.cache[cache_idx]
        self.stack.append((cache_idx, TransitionSystemVertex(cache_word_idx, cache_symbol_idx)))

    def __str__(self):
        ret = "Word Index Buffer: %s" % str(self.widx_buffer)
        ret += "  Cache: %s" % str(self.cache)
        ret += "  Stack: %s" % str(self.stack)
        return ret

    # dump the constructed hypothesis in current configuration.
    def toString(self, check=False, unreserve_gen_method=UnreserveGenMethod.NONE):
        if check:
            assert len(self.symbolSeq) == self.hypothesis.counter, \
                    "len(self.symbolSeq): {}\nself.hypothesis.counter: {}\nself.symbolSeq: {}\n".format(
                            len(self.symbolSeq), self.hypothesis.counter, self.symbolSeq)
            assert len(self.symbolSeq) == len(self.hypothesis.symbols), \
                    "len(self.symbolSeq): {}\nlen(self.hypothesis.symbols): {}\nself.symbolSeq: {}\nself.hypothesis.symbols: {}\n".format(
                            len(self.symbolSeq), len(self.hypothesis.symbols), self.symbolSeq, self.hypothesis.symbols)

        symbol_line_reprs = self.toConll(unreserve_gen_method=unreserve_gen_method)
        # NB: ulf_type_fn should be memoized so this doesn't repeat computations.
        atomic_types = [self.ulf_type_fn(con.getValue()) for con in self.hypothesis.symbols]
        symbol_line_reprs = [
            "\t".join([str(elem) for elem in [sidx, symbol_l, rel_str, par_str, atom_type_str, type_str]])
            for (sidx, ((symbol_l, rel_str, par_str), atom_type_str, type_str))
            in enumerate(zip(symbol_line_reprs, atomic_types, self.sidx2type))
        ]
        return "\n".join(symbol_line_reprs)

    def toConll(self, unreserve_gen_method=UnreserveGenMethod.NONE):
        symbol_line_reprs = []
        #print("symbolSeq: {}".format(self.symbolSeq))
        #print("hypothesis.symbols: {}".format([x.getValue() for x in self.hypothesis.symbols]))
        if unreserve_gen_method == UnreserveGenMethod.NONE and len(self.hypothesis.symbols) != len(self.symbolSeq):
            print("WARNING: generated ULFAMRGraph symbol length ({}), doesn't "
                    "match the original symbol sequence: {}".format(
                        len(self.hypothesis.symbols),
                        len(self.symbolSeq),
                    )
            )
        for (sidx, curr_symbol) in enumerate(self.hypothesis.symbols):
            #print("sidx: {}".format(sidx))
            #print("curr_symbol.getValue(): {}".format(curr_symbol.getValue()))
            curr_symbol.rebuild_ops()
            symbol_l = curr_symbol.getValue()
            #symbol_l = self.symbolSeq[sidx]
            #assert curr_symbol.getValue() == symbol_l, "curr_symbol.getValue(): {}\tsymbol_l: {}".format(curr_symbol.getValue(), symbol_l)

            # Then relations
            rel_str = "#".join([
                "%s:%d" % (r_l, curr_symbol.tail_ids[r_idx])
                for (r_idx, r_l)
                in enumerate(curr_symbol.rels)
            ])
            if not rel_str:
                rel_str = "NONE"

            # Then parent relations
            par_str = "#".join([
                "%s:%d" % (p_l, curr_symbol.parent_ids[p_idx])
                for (p_idx, p_l)
                in enumerate(curr_symbol.parent_rels)
            ])
            if not par_str:
                par_str = "NONE"

            # symbol_repr = "%d\t%s\t%s\t%s" % (sidx, symbol_l, rel_str, par_str)
            symbol_line_reprs.append((symbol_l, rel_str, par_str))
        return symbol_line_reprs


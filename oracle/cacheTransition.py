import sys, os
import string
import json
from .cacheConfiguration import *
from lisp.composition import ulf_lib
from .utils import (
    ArcDir,
    FeatureType,
    FocusMethod, FOCUS_NAMES,
    ReserveGenMethod, RESERVE_GEN_NAMES,
    TypeCheckingMethod, TYPE_CHECKING_NAMES,
    UnreserveGenMethod, UNRESERVE_GEN_NAMES,
    NONE_WIDX, NULL_WIDX, NULL_SIDX,
)
from .ulf_gen import split_ulf_atom
from .focus import FocusManager
from .exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()

NULL = "-NULL-"
UNKNOWN = "-UNK-"
RM_PUNCT_TRANS = str.maketrans('', '', string.punctuation)
# Punctuation that may participate in the WORDGEN action.
WORDGEN_PUNCTS = [
    '!',
    '?',
]
# Logicals symbols that correspond to lexical entries in the surface text.
# These are the only non-numerical symbols that are allowed to be generated
# from tokens or lemmas without any extensions.
LEXICO_LOGICAL_SYMS = [
    'to',
    'that',
    'not',
    'if',
    'whether',
    '\'s',
    '?',
    '!',
]

def is_lexico_logical_sym(string):
    """Returns whether the given string is a lexico-logical symbol. That is,
    a logical symbol in ULF that corresponds to a surface string in English.
    These can be numbers.
    """
    return string.lower() in LEXICO_LOGICAL_SYMS or \
            string.replace('.', '', 1).isdigit() # Digits with up to one decimal


class CacheTransition(object):
    def __init__(self,
            size,
            sym_labs_=None,
            arc_labs_=None,
            inseq_symbols=None,
            promote_symbols=None,
            reserve_gen_method=ReserveGenMethod.NONE,
            unreserve_gen_method=UnreserveGenMethod.NONE,
            focus_method=FocusMethod.NONE,
            type_checking_method=TypeCheckingMethod.NONE,
            ulf_lexicon=None,
            buffer_offset=0,
            ):
        self.sym_labs = sym_labs_
        self.arc_labs = arc_labs_
        self.cache_size = size
        self.push_actions = []
        self.symID_actions = []
        self.symGen_actions = []
        self.arc_actions = []
        self.gensym_actions = []
        self.push_actionIDs = dict()
        self.symID_actionIDs = dict()
        self.symGen_actionIDs = dict()
        self.arc_actionIDs = dict()
        self.gensym_actionIDs = dict()

        # Mapping from action_type to action lists.
        self.__type2actions = [
            self.symID_actions,
            self.arc_actions,
            self.push_actions,
            self.gensym_actions,
        ]

        self.outgo_arcChoices = None
        self.income_arcChoices = None
        self.default_arcChoices = None

        self.push_action_set = None
        self.arcbinary_action_set = None
        self.arclabel_action_set = None
        self.shiftpop_action_set = None
        self.gensym_action_set = None
        self.seqgen_action_set = None
        self.promote_sym_action_set = None
        self.promote_arc_action_set = None

        # buffer index offset for special tokens (e.g. beginning of sequence)
        self.buffer_offset = buffer_offset

        # for TypeCheckingMethod.COMPOSITION type checking
        self.directional_compose_types = ulf_lib.directional_compose_types

        # Generating unaligned symbols.
        self.inseq_syms = set(inseq_symbols) if inseq_symbols is not None else set()
        self.promote_syms = set(promote_symbols) if promote_symbols is not None else set()
        self.rsyms = self.inseq_syms | self.promote_syms

        # reserve_gen_method, unreserve_reserve_gen_method, and
        # type_checking_method can either be configuration strings
        # corresponding to Enum values or the Enum values themselves.
        if type(reserve_gen_method) == str:
            if reserve_gen_method.lower() in RESERVE_GEN_NAMES:
                self.reserve_gen_method = RESERVE_GEN_NAMES[reserve_gen_method.lower()]
            else:
                raise Exception("Unknown reserve gen method string: {}".format(reserve_gen_method))
        else:
            self.reserve_gen_method = reserve_gen_method
        if type(unreserve_gen_method) == str:
            if unreserve_gen_method.lower() in UNRESERVE_GEN_NAMES:
                self.unreserve_gen_method = UNRESERVE_GEN_NAMES[unreserve_gen_method.lower()]
            else:
                raise Exception("Unknown unreserve gen method string: {}".format(unreserve_gen_method))
        else:
            self.unreserve_gen_method = unreserve_gen_method
        if type(focus_method) == str:
            if focus_method.lower() in FOCUS_NAMES:
                self.focus_method = FOCUS_NAMES[focus_method.lower()]
            else:
                raise Exception("Unknown type checking method string: {}".format(focus_method))
        else:
            self.focus_method = focus_method

        if type(type_checking_method) == str:
            if type_checking_method.lower() in TYPE_CHECKING_NAMES:
                self.type_checking_method = TYPE_CHECKING_NAMES[type_checking_method.lower()]
            else:
                raise Exception("Unknown type checking method string: {}".format(type_checking_method))
        else:
            self.type_checking_method = type_checking_method

        self.lexicon = ulf_lexicon
        self.focus_manager = FocusManager.from_transition_system(self)


    # return the number of specific actions appending
    def actionNum(self, action_type):
        return len(self.transitionActions(action_type))

    # Return all possible arc choices for a certain symbol?
    def constructActionSet(self, left_symbol, right_symbol):
        # What is const?
        def is_const(s):
            const_set = set(['interrogative', 'imperative', 'expressive', '-'])
            return s in const_set or s == "NUMBER"
        label_candidates = set()
        # Take in symbol on left and right of current word
        if left_symbol in self.outgo_arcChoices and right_symbol in self.income_arcChoices:
            label_candidates |= set(["R-%s" % l for l in (
                self.outgo_arcChoices[left_symbol] & self.income_arcChoices[right_symbol])])
        if right_symbol in self.outgo_arcChoices and left_symbol in self.income_arcChoices:
            label_candidates |= set(["L-%s" % l for l in (
                self.outgo_arcChoices[right_symbol] & self.income_arcChoices[left_symbol])])
        if len(label_candidates) == 0:
            if is_const(left_symbol):
                return set(["L-%s" % l for l in self.income_arcChoices[left_symbol]])
            if is_const(right_symbol):
                return set(["R-%s" % l for l in self.income_arcChoices[right_symbol]])
            #print>> sys.stderr, "Left symbol: %s, Right symbol: %s" % (left_symbol, right_symbol)
            return self.default_arcChoices
        return label_candidates

    # Return the action sequence
    def transitionActions(self, action_type):
        return self.__type2actions[action_type]

    # Realize what action is called
    def actionStr(self, action_type, action_idx):
        return self.transitionActions(action_type)[action_idx]

    def isTerminal(self, c: CacheConfiguration, with_gold=True):
        # The coverage check is irrelevant if unreserve_gen_method == UnreserveGenMethod.WORD since the
        # symbol sequences are filled dynamically. The counter might be up to the sequence at any point.
        coverage_check = (
                c.hypothesis.counter == len(c.gold.symbols)
                if with_gold
                else ((c.hypothesis.u_counter == len(c.symbolSeq)) or
                    (self.unreserve_gen_method == UnreserveGenMethod.WORD))
        )
        # Don't require all words to be processed if using the unreserve_gen_method.
        # It may have skipped a few words.
        widx_buffer_check = c.widxBufferSize() == 0
        # Must be in the ARCBINARY phase (i.e. right after a POP)
        phase_check = c.phase == FeatureType.ARCBINARY
        # Cache must be empty.
        cache_check = all([
            (vertex.word_idx == NULL_WIDX and vertex.symbol_idx == NULL_SIDX)
            for vertex in c.cache
        ])
        # Last action was a POP
        last_action_check = c.last_action == "POP"
        #print("c.stackSize(): {}".format(c.stackSize()))
        #print("coverage_check: {}".format(coverage_check))
        #print("widx_buffer_check: {}".format(widx_buffer_check))
        #print("phase: {}".format(c.phase))
        #print("phase_check: {}".format(phase_check))
        #print("cache_check: {}".format(cache_check))
        #print()
        return (c.stackSize() == 0 and coverage_check and widx_buffer_check and \
                phase_check and cache_check and last_action_check)

    # Important method
    # Define all transition actions
    def makeTransitions(self):
        """
        Construct each type of transition actions.
        :param symbol_labels:
        :param arc_labels:
        :return:
        """
        for l in self.sym_labs:
            curr_action = "symID:" + l
            # symID_actionIDs a dictionary, utilize curr_action string as
            # key and len tells this is the n th action
            self.symID_actionIDs[curr_action] = len(self.symID_actions)
            self.symID_actions.append(curr_action)

            gen_action = "symGen:" + l
            self.symGen_actionIDs[gen_action] = len(self.symGen_actions)
            self.symGen_actions.append(gen_action)

        null_action = "symID:" + NULL
        self.symID_actionIDs[null_action] = len(self.symID_actions)
        self.symID_actions.append(null_action)
        null_gen = "symGen:" + NULL
        self.symGen_actionIDs[null_gen] = len(self.symGen_actions)
        self.symGen_actions.append(null_gen)

        unk_action = "symID:" + UNKNOWN
        self.symID_actionIDs[unk_action] = len(self.symID_actions)
        self.symID_actions.append(unk_action)
        unk_gen = "symGen:" + UNKNOWN
        self.symGen_actionIDs[unk_gen] = len(self.symGen_actions)
        self.symGen_actions.append(unk_gen)

        for l in self.arc_labs:
            if l == NULL or l == UNKNOWN:
                continue
            curr_action = "L-" + l
            self.arc_actionIDs[curr_action] = len(self.arc_actions)
            self.arc_actions.append(curr_action)

            curr_action = "R-" + l
            self.arc_actionIDs[curr_action] = len(self.arc_actions)
            self.arc_actions.append(curr_action)

        for i in range(self.cache_size):
            curr_action = "PUSH:%d" % i
            self.push_actionIDs[curr_action] = len(self.push_actions)
            self.push_actions.append(curr_action)

        for sym in self.inseq_syms:
            curr_action = "symGen:" + sym
            self.gensym_actionIDs[curr_action] = len(self.gensym_actions)
            self.gensym_actions.append(curr_action)

    def getActionIDX(self, action, type):
        if type == 0:
            return self.symID_actionIDs[action]
        elif type == 1:
            return self.arc_actionIDs[action]
        elif type == 2:
            return self.push_actionIDs[action]
        elif type == 3:
            return self.gensym_actionIDs[action]
        return -1

    # helper method for composition checking
    def canApplyArc(self, c, cache_idx, symbol_idx, direction):
        return self.typeCheck(c, cache_idx, symbol_idx, direction) and self.bottomupCheck(c, symbol_idx, cache_idx)

    def typeCheck(self, c, cache_idx, symbol_idx, direction):
        """Helper method for ULF type compositino checking.
        Checks that the ARC is allowed by the Lisp ULF composition and returns
        the composed type.
        """
        if symbol_idx < 0 or cache_idx < 0:
            return False
        #print("c.sidx2type: {}".format(c.sidx2type))
        #print("symbol_idx: {}".format(symbol_idx))
        #print("cache_idx: {}".format(cache_idx))
        # already have a type for all symbols that are aligned to word
        #cur_symbol_seq = c.hypothesis.symbols
        #query_symbol = cur_symbol_seq[symbol_idx - 1]  # symbol_idx is next shifted index
        #cache_symbol = cur_symbol_seq[cache_idx]
        type1 = c.sidx2type[cache_idx]
        type2 = c.sidx2type[symbol_idx]  # symbol_idx is next shifted index
        if type1 == '...' or type2 == '...':
            return False
        dirstr = "left" if direction == ArcDir.LEFT else "right"
        next_compose_type = self.directional_compose_types(type1, type2, dirstr)
        #print("type checking...\nsymbol_idx: {}\ncache_idx: {}\ntype1: {}\ntype2: {}\ncompose_type: {}".format(symbol_idx, cache_idx, type1, type2, next_compose_type))
        if next_compose_type == '...':
            return False
        c.next_compose_type = next_compose_type
        return True

    # simply check to see if the symbol already makes a parent arc
    # if the child already has a parent or the parent already has a parent
    # the arc cannot be made
    def bottomupCheck(self, c, symbol_idx, cache_idx):
        #return True
        return c.symbol_available[symbol_idx] and c.symbol_available[cache_idx]

    #helper function to check if a node on path connecting to something being processed
    def connectStackOrCache(self, c, vertex_idx, parent_only=False):
        # TODO: add a check method to see if is root, if so, no need and default return
        # True since process completed
        for parent_idx in c.hypothesis.symbols[vertex_idx].parent_ids:
            if parent_idx in c.stack:
                return True
            for (_, cache_vertex_idx) in c.cache:
                if cache_vertex_idx == vertex_idx:
                    continue
                if parent_idx == vertex_idx:
                    return True
        if not parent_only:
            for parent_idx in c.hypothesis.symbols[vertex_idx].parent_ids:
                if parent_idx in c.stack:
                    return True
                for (_, cache_vertex_idx) in c.cache:
                    if cache_vertex_idx == vertex_idx:
                        continue
                    if parent_idx == vertex_idx:
                        return True
                # check if the current parent, although itself not on stack
                # or cache, is on the path connecting back to stack or cache
                if self.connectStackOrCache(c, parent_idx):
                    return True

            for child_idx in c.hypothesis.symbols[vertex_idx].tail_ids:
                if child_idx in c.stack:
                    return True
                for (_, cache_vertex_idx) in c.cache:
                    if cache_vertex_idx == vertex_idx:
                        continue
                    if child_idx == vertex_idx:
                        return True

                if self.connectStackOrCache(c, child_idx):
                    return True

        #if neither children nor parents being processed, disjoint
        return False

    def checkHasParent(self, c, vertex_idx):
        # Simple check if the given vertex has a parent edge.
        return vertex_idx == 0 or len(c.hypothesis.symbols[vertex_idx].parent_ids) != 0

    def canPop(self, c, bottom_up=False):
        _, cache_vertex_idx = c.cache[-1]

        return self.checkHasParent(c, cache_vertex_idx)
        # if bottom up only need to check for immediate parent
        #if bottom_up:
        #    return self.connectStackOrCache(c, cache_vertex_idx, True)
        #else:
        #    return self.connectStackOrCache(c, cache_vertex_idx)

    def canApply(self, c, action, use_refined=False, pop_refined=False, cache_idx=-1, no_trailing_seqgen=False):
        """Returns whether `action` can be applied to cache transition state `c`.
        Args:
            c: Cache transition state, CacheConfiguration class.
            action: String representation of transition action.
            use_refined: Whether to use the type system for more refined
                applicability decisions.
            pop_refined: Whether the pop action should restrict parent-child
                relation orderings.
            cache_idx: Current cache index. Only used in ARC actions.
            no_trailing_seqgen: flag to stop SEQGEN actions after all input words have been processed
        """
        if self.unreserve_gen_method == UnreserveGenMethod.NONE:
            return self.origCanApply(c, action, use_refined, pop_refined, cache_idx)
        elif self.unreserve_gen_method == UnreserveGenMethod.WORD:
            return self.wordGenCanApply(c, action, use_refined, pop_refined, cache_idx, no_trailing_seqgen)
        else:
            raise Exception("Unknown can apply variant")

    def origCanApply(self, c, action, use_refined=False, pop_refined=False, cache_idx=-1):
        """Returns whether `action` can be applied to cache transition state `c`.
        Args:
            c: Cache transition state, CacheConfiguration class.
            action: String representation of transition action.
            use_refined: Whether to use the type system for more refined
                applicability decisions.
            pop_refined: Whether the pop action should restrict parent-child
                relation orderings.
            cache_idx: Current cache index. Only used in ARC actions.
        """
        #print("In origCanApply")
        #print("c.phase: {}".format(c.phase))
        #print("action: {}".format(action))
        #print("unreserve_gen_method: {}".format(self.unreserve_gen_method))

        _, symbol_idx = self.focus_manager.get(c)
        if c.phase == FeatureType.SHIFTPOP: # Can shift, pop, mergebuf, or gensym/seqgen.
            if action not in self.shiftpop_action_set:
                return False
            if (self.unreserve_gen_method == UnreserveGenMethod.NONE) and \
                    (c.getSymbol(symbol_idx, orig=True) is None):
                # at the end (e.g. *h, *p, {you}.pro, etc.). Currently not allowing
                # symbols at the end because the parser overgenerates if we allow it.
                #return (action == "POP" and c.stackSize()) or action == "SEQGEN"
                return action == "POP" and c.stackSize()
            if c.stackSize() < 1:
                return action != "POP"
            if pop_refined and action == "POP":
                # if use type system, restrict to bottom up
                bottom_up = True # assume bottom up for now
                return self.canPop(c, bottom_up)
            if (action == "SHIFT") and (c.getSymbol(symbol_idx, orig=True) is None):
                return False
            return True
        if c.phase == FeatureType.PUSHIDX:
            return action in self.push_action_set
        if c.phase == FeatureType.ARCBINARY:
            _, left_symbol_idx = c.getCache(cache_idx)
            if left_symbol_idx == NULL_SIDX or left_symbol_idx is None:
                return action == "NOARC"

            # if no need for checking type and bottom up issue, allows as long as valid action
            # else if action is NOARC, always applicable
            # otherwise, check if direction, type and bottom up issues checked
            if self.type_checking_method != TypeCheckingMethod.COMPOSITION:
                return action in self.arcbinary_action_set
            elif action == "NOARC":
                return True
            else:
                # COMPOSITION check. NB: the SANITY_CHECK is performed at the Hypothesis stage.
                action_arc_label = action.split(":")[1][0]
                if action_arc_label == "L":
                    return self.canApplyArc(c, left_symbol_idx, symbol_idx, 0) and action in self.arcbinary_action_set
                else:
                    return self.canApplyArc(c, left_symbol_idx, symbol_idx, 1) and action in self.arcbinary_action_set
        if c.phase == FeatureType.SYMSELECT:
            return "seqGen" in action

        if c.phase == FeatureType.WORDGEN:
            return action in ["LEMMA", "NAME", "TOKEN"]
        if c.phase in [
                FeatureType.WORDGEN_LEMMA,
                FeatureType.WORDGEN_NAME,
                FeatureType.WORDGEN_TOKEN
                ]:
            return "SUFFIX" in action

        if use_refined:
            left_symbol = c.getCacheSymbol(cache_idx, utils.TokenType.SYMBOL)
            right_symbol = c.getCacheSymbol(self.cache_size-1, utils.TokenType.SYMBOL)
            arc_choices = self.constructActionSet(left_symbol, right_symbol)
            connected = c.getConnectedArcs(cache_idx) | c.getConnectedArcs(self.cache_size-1, False)
            return action in arc_choices and action not in connected
        return action in self.arclabel_action_set


    def wordGenCanApply(self, c, action, use_refined=False, pop_refined=False, cache_idx=-1, no_trailing_seqgen=False):
        """Returns whether `action` can be applied to cache transition state `c`, specifically for
        a transition system that generates symbols from the words.
        Args:
            c: Cache transition state, CacheConfiguration class.
            action: String representation of transition action.
            use_refined: Whether to use the type system for more refined
                applicability decisions.
            pop_refined: Whether the pop action should restrict parent-child
                relation orderings.
            cache_idx: Current cache index. Only used in ARC actions.
            no_trailing_seqgen: flag to stop SEQGEN actions after all input words have been processed
        """

        #print("In wordGenCanApply")
        #print("c.phase: {}".format(c.phase))
        #print("action: {}".format(action))
        _, symbol_idx = self.focus_manager.get(c)
        if c.phase == FeatureType.SHIFTPOP: # Can shift, pop, mergebuf, or gensym/seqgen.
            if action not in self.shiftpop_action_set:
                return False
            if (self.unreserve_gen_method == UnreserveGenMethod.NONE) and \
                    (c.getSymbol(symbol_idx, orig=True) is None):
                return action == "POP" and c.stackSize() > 0
            if c.stackSize() < 1 and action == "POP":
                return False
            if pop_refined and action == "POP":
                # if use type system, restrict to bottom up
                return self.canPop(c, use_refined)
            if (action == "SHIFT") and (c.getSymbol(symbol_idx, orig=True) is None):
                return False
            if action == "WORDGEN":
                # WORDGEN must have an available word, and not allowed on punctuation other than WORDGEN_PUNCTS
                if c.peekWordIdx() >= 0:
                    cur_tok = c.getCurToken()
                    return cur_tok.translate(RM_PUNCT_TRANS) != '' or cur_tok in WORDGEN_PUNCTS
                else:
                    return False
            if action == "symID:-NULL-":
                # Can't skip a token if we've done some merging already.
                # Also can't skip a token if there are no words left.
                return c.widx_width == 1 and c.peekWordIdx() >= 0
            if c.peekWordIdx() < 0:
                if no_trailing_seqgen:
                    return action not in ["WORDGEN", "MERGEBUF", "symID:-NULL-", "SEQGEN"]
                else:
                    return action not in ["WORDGEN", "MERGEBUF", "symID:-NULL-"]
            if action == "MERGEBUF":
                return c.widx_width < c.widxBufferSize() - 1
            return True
        if c.phase == FeatureType.PUSHIDX:
            return action in self.push_action_set
        if c.phase == FeatureType.ARCBINARY:
            _, left_symbol_idx = c.getCache(cache_idx)
            if left_symbol_idx == NULL_SIDX or left_symbol_idx is None:
                return action == "NOARC"

            # if no need for checking type and bottom up issue, allows as long as valid action
            # else if action is NOARC, always applicable
            # otherwise, check if direction, type and bottom up issues checked
            return action in self.arcbinary_action_set
        if c.phase == FeatureType.ARCCONNECT:
            _, left_symbol_idx = c.getCache(cache_idx)
            _, right_symbol_idx = c.rightmostCache()
            act_split = action.split("-")
            if left_symbol_idx == NULL_SIDX or left_symbol_idx is None or act_split[0] not in ["L", "R"] or len(act_split) < 2:
                return False
            if self.type_checking_method != TypeCheckingMethod.COMPOSITION:
                return self.bottomupCheck(c, right_symbol_idx, left_symbol_idx)
            action_dir = ArcDir.LEFT if act_split[0] == "L" else ArcDir.RIGHT
            return (action in self.arclabel_action_set and
                    self.canApplyArc(c, left_symbol_idx, right_symbol_idx, action_dir))
        if c.phase == FeatureType.SYMSELECT:
            return "seqGen" in action

        if c.phase == FeatureType.WORDGEN:
            return action in ["LEMMA", "NAME", "TOKEN"]
        if c.phase == FeatureType.WORDGEN_NAME:
            # Names can't be suffix-less (those are Pronoun)
            if "SUFFIX:" not in action or action.endswith(":None"):
                return False
            # If a lexicon is available check the current action.
            #             common nouns to participate in names if necessary.
            if self.lexicon is not None:
                return self.lexicon.check_name_action(c, action, strict_override=False)
            return True
        if c.phase in [
                FeatureType.WORDGEN_LEMMA,
                FeatureType.WORDGEN_TOKEN
                ]:
            if "SUFFIX:" not in action:
                return False
            # Lemma and Token generation without a suffix is only allowed for a few reserved symbols.
            if action == "SUFFIX:None":
                return is_lexico_logical_sym(c.getCurWord())
            # If a lexicon is available check the current action.
            if self.lexicon is not None:
                return self.lexicon.check_action(c, action)
            return True

        if c.phase == FeatureType.PROMOTE:
            # Rightmost cache element must not be non-null for a PROMOTE action.
            _, sidx = c.rightmostCache()
            #if action != "NOPROMOTE":
            #    assert False
            return (sidx != NULL_SIDX and action == "PROMOTE") or action == "NOPROMOTE"
        if c.phase == FeatureType.PROMOTE_SYM:
            # Check that the symbol is in the restricted symbol set.
            if not action.startswith("PROMOTE_SYM:"):
                return False
            sym = action.split(":")[1]
            #assert False
            return sym in self.promote_syms
        if c.phase == FeatureType.PROMOTE_ARC:
            if not action.startswith("PROMOTE_ARC:"):
                return False
            _, cache_sidx = c.rightmostCache()
            # Subtract one from index since we want the last symbol generated
            promote_symbol_idx = c.next_symbol_data.all - 1
            if self.type_checking_method != TypeCheckingMethod.COMPOSITION:
                return self.bottomupCheck(c, promote_symbol_idx, cache_sidx)
            # Check arc composition
            return (action in self.promote_arc_action_set and
                    self.canApplyArc(c, cache_sidx, promote_symbol_idx, ArcDir.LEFT))

        if use_refined:
            left_symbol = c.getCacheSymbol(cache_idx, utils.TokenType.SYMBOL)
            right_symbol = c.getCacheSymbol(self.cache_size-1, utils.TokenType.SYMBOL)
            arc_choices = self.constructActionSet(left_symbol, right_symbol)
            connected = c.getConnectedArcs(cache_idx) | c.getConnectedArcs(self.cache_size-1, False)
            return action in arc_choices and action not in connected
        return action in self.arclabel_action_set


    def apply(self, c, action):
        """Applies `action`, generated from the oracle or the parser, to the
        transition system configuration, `c`.
        NB: The actions here will be different from the actions from canApply
        since this takes a compacted version of the canApply actions, designed
        just for the need to the oracle. canApply uses actions that are
        simplified for the transition system.
        """
        c.next_symbol_data = self.focus_manager.update_symbol_data(
                c.next_symbol_data, action)
        if action == "POP": # POP action
            if not c.pop():
                assert False, "Pop from empty stack!"
            c.start_word = False
            c.last_action = "POP"
            c.phase = FeatureType.ARCBINARY

        elif action == "NOPROMOTE":
            # Skip promote stage, just go to shiftpop.
            c.start_word = True
            c.last_action = "NOPROMOTE"
            c.phase = FeatureType.SHIFTPOP

        elif "PROMOTE_SYM:" in action:
            # First, add the new symbol to the hypothesis.
            _, symbol = action.split(":")
            promote_symbol = SymbolLabel(symbol)
            sidx, _, _ = c.hypothesis.addSymbol(promote_symbol, restricted=True)
            if self.type_checking_method == TypeCheckingMethod.COMPOSITION:
                # Compute and add the ulf type.
                c.addAtomicULFType(sidx)
            if c.symbol_available is not None:
                # Add a new available symbol.
                c.symbol_available.append(True)
            # Set configuration state.
            # Go back to arcs in case we want arcs with this promoted element.
            c.phase = FeatureType.PROMOTE_ARC
            c.start_word = False
            c.pop_widx_buff = False
            c.last_action = "PROMOTE_SYM"

        elif "PROMOTE_ARC:" in action:
            assert not c.start_word
            _, arclabel = action.split(":")
            # Make new transition system vertex and connect with rightmost cache.
            _, curr_cache_symbol_idx = c.rightmostCache()
            promote_symbol_idx = c.hypothesis.nextSymbolIDX()
            c.connectArc(
                curr_cache_symbol_idx,
                promote_symbol_idx,
                ArcDir.LEFT,
                arclabel,
                use_types=(self.type_checking_method == TypeCheckingMethod.COMPOSITION),
            )
            # Pop out current cache element and push in promoted one.
            c.cand_vertex = TransitionSystemVertex(NONE_WIDX, promote_symbol_idx)
            c.cache.pop()
            c.moveToCache(c.cand_vertex)
            c.hypothesis.count(restricted=c.hypothesis.isRestrictedIdx(c.cand_vertex.symbol_idx))
            # Set configuration state.
            # Go back to arcs in case we want arcs with this promoted element.
            c.phase = FeatureType.ARCBINARY
            c.start_word = False
            c.pop_widx_buff = False
            c.last_action = "PROMOTE_ARC"

        elif action == "symID:-NULL-":
            # Shift past word without generating symbol.
            c.popWidxBuffer()
            c.start_word = True
            c.phase = FeatureType.SHIFTPOP
            c.last_action = action

        elif action == "symID:-SKIP-":
            # Skip this SHIFTPOP stage.
            # Cache items aren't ready to be popped and there's nothing to generate.
            c.start_word = False
            c.phase = FeatureType.ARCBINARY
            c.last_action = action

        elif "mergeBuf" is action:
            # Merge next two pieces of the buffer and retry generating symbol.
            c.start_word = True
            c.widx_width += 1
            c.phase = FeatureType.SHIFTPOP

        elif "symID" in action or "symMod" in action:
            # Get the symbol name/the symbol ID
            #print("in symID/symMod action: {}".format(action))
            l = action.split(":")[1]
            new_symbol = SymbolLabel(l)
            new_hypo_symbol_idx = c.hypothesis.nextSymbolIDX()
            sidx, _, _ = c.hypothesis.addSymbol(new_symbol, restricted=False)
            if self.type_checking_method == TypeCheckingMethod.COMPOSITION:
                # Compute and add the ulf type.
                c.addAtomicULFType(sidx)
            if c.symbol_available is not None:
                # Add a new available symbol.
                c.symbol_available.append(True)
            c.phase = FeatureType.PUSHIDX
            c.start_word = False # Have generated a candidate vertex.
            next_word = c.peekWordIdx()
            c.cand_vertex = TransitionSystemVertex(next_word, new_hypo_symbol_idx)
            c.pop_widx_buff = True
            c.last_action = "symID"

        elif "lemmaGen" in action or "tokenGen" in action:
            # Generate symbol from the lemma or token and given suffix.
            segments = action.split(":")
            word = segments[1]
            suffix = segments[2]
            # Numbers with suffixes get strings and bars generated around them.
            if word.replace(".", "", 1).isdigit() and suffix != "None":
                new_symbol = SymbolLabel("\"|{}.{}|\"".format(word, suffix).upper())
            elif suffix == "None":
                new_symbol = SymbolLabel(word)
            else:
                new_symbol = SymbolLabel("{}.{}".format(word, suffix).upper())
            new_hypo_symbol_idx = c.hypothesis.nextSymbolIDX()
            sidx, _, _ = c.hypothesis.addSymbol(new_symbol, restricted=False)
            if self.type_checking_method == TypeCheckingMethod.COMPOSITION:
                # Compute and add the ulf type.
                c.addAtomicULFType(sidx)
            if c.symbol_available is not None:
                # Add a new available symbol.
                c.symbol_available.append(True)
            c.phase = FeatureType.PUSHIDX
            c.cand_vertex = TransitionSystemVertex(c.peekWordIdx(), new_hypo_symbol_idx)
            c.start_word = False
            c.pop_widx_buff = True
            c.last_action = action.split(":")[0]

        elif "nameGen" in action:
            # Generate a named symbol from string and given suffix.
            # Generate symbol from the lemma and given suffix.
            segments = action.split(":")
            tokens = segments[1].replace("<-SPACE->", " ")
            suffix = segments[2]
            new_symbol = SymbolLabel("\"| {}.{}|\"".format(tokens, suffix))
            if tokens == "\"|\\\"|\"":
                new_symbol = SymbolLabel(tokens)
            elif suffix.upper() == "PRO":
                new_symbol = SymbolLabel("\"| {}|\"".format(tokens))
            new_hypo_symbol_idx = c.hypothesis.nextSymbolIDX()
            sidx, _, _ = c.hypothesis.addSymbol(new_symbol, restricted=False)
            if self.type_checking_method == TypeCheckingMethod.COMPOSITION:
                # Compute and add the ulf type.
                c.addAtomicULFType(sidx)
            if c.symbol_available is not None:
                # Add a new available symbol.
                c.symbol_available.append(True)
            c.phase = FeatureType.PUSHIDX
            c.cand_vertex = TransitionSystemVertex(c.peekWordIdx(), new_hypo_symbol_idx)
            c.start_word = False
            c.pop_widx_buff = True
            c.last_action = "nameGen"

        elif "symGen" in action:
            #print("in symGen action: {}".format(action))
            l = action.split(":")[1]
            new_symbol = SymbolLabel(l)
            new_symbol_idx = c.hypothesis.nextSymbolIDX()
            sidx, _, _ = c.hypothesis.addSymbol(
                    new_symbol,
                    restricted=(l in self.rsyms)
            )
            if self.type_checking_method == TypeCheckingMethod.COMPOSITION:
                # Compute and add the ulf type.
                c.addAtomicULFType(sidx)
            if c.symbol_available is not None:
                # Add a new available symbol.
                c.symbol_available.append(True)
            c.cand_vertex = TransitionSystemVertex(NONE_WIDX, new_symbol_idx)
            c.phase = FeatureType.PUSHIDX
            c.start_word = False
            c.pop_widx_buff = False
            c.last_action = "symGen"

        elif "seqGen" in action:
            # Generate restricted symbol in sequence.
            #print("in seqGen action: {}".format(action))
            l = action.split(":")[1]
            new_symbol = SymbolLabel(l)
            new_symbol_idx = c.hypothesis.nextSymbolIDX()
            sidx, _, _ = c.hypothesis.addSymbol(new_symbol, restricted=True)
            if self.type_checking_method == TypeCheckingMethod.COMPOSITION:
                # Compute and add the ulf type.
                c.addAtomicULFType(sidx)
            if c.symbol_available is not None:
                # Add a new available symbol.
                c.symbol_available.append(True)
            c.cand_vertex = TransitionSystemVertex(NONE_WIDX, new_symbol_idx)  # unaligned vertex
            c.phase = FeatureType.PUSHIDX  # push this into cache
            c.start_word = False  # next stage is not a word generation stage
            c.pop_widx_buff = False  # push action without corresponding word
            c.last_action = "seqGen"

        elif "ARC" in action:
            #print("ARC action: {}".format(action))
            assert not c.start_word

            parts = action.split(":")
            cache_idx = int(parts[0][3:])
            arc_label = parts[1]
            _, curr_cache_symbol_idx = c.getCache(cache_idx)
            if cache_idx == 0:
                if self.reserve_gen_method == ReserveGenMethod.PROMOTE:
                    c.phase = FeatureType.PROMOTE
                else:
                    c.phase = FeatureType.SHIFTPOP
                    c.start_word = True
            else:
                c.phase = FeatureType.ARCBINARY

            if arc_label == "O":
                c.last_action = "ARC"
                return
            # Apply left or right arc
            c.connectArc(
                curr_cache_symbol_idx,
                c.cand_vertex.symbol_idx,
                (ArcDir.LEFT if arc_label[0] == "L" else ArcDir.RIGHT),
                arc_label[2:],
                use_types=(self.type_checking_method == TypeCheckingMethod.COMPOSITION),
            )
            c.last_action = "ARC"

        else: # PUSHx

            #print()
            #print("push action: {}".format(action))
            #print("symbol_idx: {}".format(c.cand_vertex.symbol_idx))
            #print("hypo symbols: {}".format([con.getValue() for con in c.hypothesis.symbols]))
            #print("restricted: {}".format(c.hypothesis.isRestrictedIdx(c.cand_vertex.symbol_idx)))
            #print()

            assert not c.start_word
            cache_idx = int(action.split(":")[1])
            c.pushToStack(cache_idx)
            c.moveToCache(c.cand_vertex)
            c.hypothesis.count(restricted=c.hypothesis.isRestrictedIdx(c.cand_vertex.symbol_idx))
            c.phase = FeatureType.ARCBINARY
            if c.pop_widx_buff:
                # Pop off word indices equivalent to the width of the current token.
                for i in range(c.widx_width):
                    c.popWidxBuffer()
                c.widx_width = 1
            c.last_action = "PUSH"

        # Update the word focus.
        c.next_word_idx = c.peekWordIdx()


    # right_edges?
    # choose the most distance vertex
    def chooseVertex(self, c, gold_right_edges):
        """Chooses the vertex in the cache to push to the stack. This will be
        the vertex whose next edge is furthest in the buffer.

        IGNORE BELOW --- NO LONGER APPLICABLE
        However, for pushing generated symbols, the cache positions are
        restricted by the symbol's relation to cache elements. If the symbol is
        a parent of a cache element, the chosen vertex is one index left of the
        child and if the symbol is a child of a cache element, the chosen
        vertex must be the index of the parent vertex.
        """
        max_dist = -1
        max_idx = -1

        # Choose last useful vertex from available choices.
        for cache_idx in range(c.cache_size):
            #print("CACHE_IDX: {}".format(cache_idx))
            cache_word_idx, hypo_cache_symbol_idx = c.getCache(cache_idx)
            curr_dist = 1000
            if hypo_cache_symbol_idx == NULL_SIDX:
                # if it is $,$
                return cache_idx

            gold_cache_symbol_idx = c.hyp2gold[hypo_cache_symbol_idx]

            #if self.reserve_gen_method == ReserveGenMethod.PROMOTE:
            #    # Symbol indices won't match between gold and hypo, so some extra computation.

            #    # If no connection to any future vertices.
            #    # If they all exist and are below current symbol index, no future indices.
            #    gold_right_idxs = gold_right_edges[gold_cache_symbol_idx]
            #    hypo_right_idxs = [
            #            c.gold2hyp[right_idx] for right_idx in gold_right_idxs
            #            if right_idx in c.gold2hyp
            #    ]
            #    if (len(hypo_right_idxs) == len(gold_right_idxs) and
            #            max(hypo_right_idxs) < next_hypo_symbol_idx):
            #        return cache_idx

            if self.reserve_gen_method == ReserveGenMethod.PROMOTE:
                next_symbol_idx = c.hypothesis.nextSymbolIDX(restricted_filtered=False)
                _, cache_symbol_idx = c.getCache(cache_idx)
                hypo_cache_symbol_idx = c.gold2hyp[gold_cache_symbol_idx]

                # Get all descendant unrestricted symbols from this symbol.
                # We must look at descendents and not just nodes since we
                # require bottom-up parsing. So some restricted symbol
                # children will be realized when their children are introduced.
                u_desc_idxs = [
                        c.gold.ftou[fidx]
                        for fidx
                        in c.gold.u_desc_fidx[gold_cache_symbol_idx]
                ]
                #print("u_desc_idxs: {}".format(u_desc_idxs))
                next_u_symbol_idx = c.hypothesis.nextSymbolIDX(restricted_filtered=True)
                # Get index for the new symbol that is being pushed.
                next_f_symbol_idx = c.hypothesis.nextSymbolIDX(restricted_filtered=False)
                gold_f_symbol_idx = c.hyp2gold[next_f_symbol_idx]
                #print("next_u_symbol_idx: {}".format(next_u_symbol_idx))
                #print("next_f_symbol_idx: {}".format(next_f_symbol_idx))
                #print("gold_f_symbol_idx: {}".format(gold_f_symbol_idx))
                #print("gold_right_edges: {}".format(gold_right_edges[gold_cache_symbol_idx]))

                # Get inseq roots of all reserved symbol constituents. Basically,
                # these are reserved symbol cases that aren't handled by other
                # cases. This is a constituent that has no path through unreserved
                # means, but will be a direct child after some promotion steps.
                all_reserve_inseq_roots = [
                        c.gold.uppermostInSeqIdx(child)
                        for child
                        in gold_right_edges[gold_cache_symbol_idx]
                        if c.gold.isAllReservedConstituent(child)
                ]
                if (gold_f_symbol_idx in gold_right_edges[gold_cache_symbol_idx] or
                        gold_f_symbol_idx in all_reserve_inseq_roots):
                    #print("RIGHT EDGE CASE")
                    # If the next symbol is a direct child of this symbol,
                    # or is part of a reserved symbol-only constituent whose
                    # root is a child of this symbol, then
                    # keep it even more than having no descendants.
                    curr_dist = -2
                elif (len(u_desc_idxs) == 0 or
                        u_desc_idxs[-1] < next_u_symbol_idx):
                    #print("NO RIGHT EDGE CASE")
                    ## If no unrestricted descendants remaining to the right of
                    ## here, we want to keep this and get rid of it as soon as
                    ## possible, so give it -1 distance.
                    #curr_dist = -1
                    # It doesn't matter so just push it into the stack?
                    return cache_idx
                else:
                    #print("GENERAL CASE")
                    # Get the minimum distance that is greater than the next generated
                    # index.
                    for u_connect_idx in u_desc_idxs:
                        if u_connect_idx >= next_u_symbol_idx:
                            curr_dist = u_connect_idx
                            break
            else:
                # If no connections to the right.
                if gold_cache_symbol_idx not in gold_right_edges:
                    return cache_idx
                # Matching symbol indices -- EASY!
                next_symbol_idx = c.hypothesis.nextSymbolIDX(restricted_filtered=False)
                _, cache_symbol_idx = c.getCache(cache_idx)
                right_edges = gold_right_edges
                if (cache_symbol_idx not in right_edges or
                        right_edges[cache_symbol_idx][-1] < next_symbol_idx):
                    # No rightward edges, so select right away.
                    return cache_idx
                for connect_idx in right_edges[cache_symbol_idx]:
                    # Get the lowest value right edge.
                    if connect_idx >= next_symbol_idx:
                        curr_dist = connect_idx
                        break

            assert curr_dist != 1000

            if curr_dist > max_dist:
                max_idx = cache_idx
                max_dist = curr_dist

        return max_idx

    def promoteSingleReservedAction(self, cache_idx, c, gold_graph, hypo_graph):
        """Tests for applicability of generating a reserved symbol for a given
        cache index. If applicable returns the action.

        If the cache element satisfies the symbol generation restrictions
         - all outgoing edges except those going to reserved symbols have been generated
         - and there is at least one edge to- or from- a reserved symbol it is
           allowed to generate
         - if generating a parent symbol, the cache element is not leftmost
         - if generating a child symbol, the cache element is not rightmost
        then add it to the front of the buffer.
        """
        cache_word_idx, hypo_cache_symbol_idx = c.getCache(cache_idx)
        leftmost = cache_idx == 0
        rightmost = cache_idx == c.cache_size - 1
        #print("hypo_cache_symbol_idx: {}".format(hypo_cache_symbol_idx))

        # Cannot generate reserved symbols from null.
        if hypo_cache_symbol_idx == NULL_SIDX:
            return None

        gold_cache_symbol_idx = c.hyp2gold[hypo_cache_symbol_idx]
        #print("gold_cache_symbol_idx: {}".format(gold_cache_symbol_idx))
        #print("gold_symbols: {}".format([con.getValue() for con in gold_graph.symbols]))
        satisfied = True
        generation_candidates = []
        unconnected_restricted_children = []
        for gold_child_idx in gold_graph.headToTail[gold_cache_symbol_idx]:
            # If there is an out-going edge to be completed but not a reserved symbol, abort.
            #print("gold_child_idx: {}".format(gold_child_idx))
            #if gold_child_idx in c.gold2hyp:
            #    print("hyp_child_idx: {}".format(c.gold2hyp[gold_child_idx]))
            #print("hypo_graph.headToTail: {}".format(hypo_graph.headToTail))
            if (gold_child_idx not in c.gold2hyp or
                    c.gold2hyp[gold_child_idx] not in hypo_graph.headToTail[hypo_cache_symbol_idx]):
                if gold_graph.symbolLabel(gold_child_idx) not in self.inseq_syms:
                    satisfied = False
                    break
                elif gold_child_idx in c.gold.generating_children[gold_cache_symbol_idx]:
                    # Track restricted children that are not connected.
                    unconnected_restricted_children.append(gold_child_idx)
            # If this index doesn't exist in the mapping yet, add to candidates.
            if gold_child_idx not in c.gold2hyp:
                generation_candidates.append(gold_child_idx)

        if satisfied:
            # - get all edges to-/from- restricted symbols, that have not been added
            # - choose out-going edges first
            cur_symbol = hypo_graph.symbols[hypo_cache_symbol_idx]
            cur_gold_symbol = gold_graph.symbols[gold_cache_symbol_idx]
            #print("cur symbol: {}".format(cur_gold_symbol.getValue()))
            assert len(cur_gold_symbol.parent_ids) == 1
            parent_gold_idx = cur_gold_symbol.parent_ids[0]
            parent_gold_symbol = gold_graph.symbols[parent_gold_idx]
            if len(generation_candidates) > 0:
                next_gold_idx = generation_candidates[0]
            elif (len(unconnected_restricted_children) == 0 and
                    parent_gold_symbol.getValue() in self.inseq_syms and
                    parent_gold_idx not in c.gold2hyp and
                    not leftmost):
                next_gold_idx = parent_gold_idx
            else:
                #print("No reserved symbols left to generate!")
                next_gold_idx = None

            if next_gold_idx is not None:
                next_symbol = gold_graph.symbolLabel(next_gold_idx)
                next_hypo_idx = hypo_graph.nextSymbolIDX()
                assert next_hypo_idx not in c.hyp2gold, \
                        "next_hypo_idx: {}\tc.hyp2gold: {}".format(next_hypo_idx, c.hyp2gold)
                assert next_gold_idx not in c.gold2hyp, \
                        "next_gold_idx: {}\tc.gold2hyp: {}".format(next_gold_idx, c.gold2hyp)
                #c.hyp2gold[next_hypo_idx] = next_gold_idx
                #c.gold2hyp[next_gold_idx] = next_hypo_idx
                c.last_action = "symGen"
                return "symGen:" + next_symbol, next_hypo_idx, next_gold_idx
        return None

    def promoteReservedAction(self, c, gold_graph, hypo_graph):
        """Tests for applicability of generating a reserved symbol, and if
        applicable returns the action. If not applicable, returns None.

        If any element in cache satisfies the symbol generation restrictions
         - all outgoing edges except those going to reserved symbols have been generated
         - and there is at least one edge to- or from- a reserved symbol
        then add it to the front of the buffer.

        Goes right-to-left in the cache so we can finish rightmost elements first.
        """
        # the promoted element replaces the cache elem.
        for cache_idx in range(c.cache_size - 1, -1, -1):
            action = self.promoteSingleReservedAction(cache_idx, c, gold_graph, hypo_graph)
            if action is not None:
                return action

    def generateReservedInSeqAction(self, c, gold_graph, hypo_graph):
        """Generates a reserved symbol in sequence.
        """
        # Since the symbol here is generated in sequence, gold index matches
        # the hypo index.
        next_symbol_idx = hypo_graph.nextSymbolIDX()
        if gold_graph.isInSeqIdx(next_symbol_idx):
            next_symbol = gold_graph.symbolLabel(next_symbol_idx)
            return "seqGen:" + next_symbol, next_symbol_idx, next_symbol_idx

    def generateReservedSymbol(self, c, gold_graph):
        """Since some symbols are reserved with no child, these symbols
        cannot be generated through promote. Therefore, conGen still
        requires in order to generate these symbols.
        This proceess is taken when the parser finishes all the process
        due to last push.
        Even if one node has multiple child of this type, generate
        one at a time. After such node is generated, as it has no child,
        it will be popped immediately. Then for the parent node, if it has
        more children of this type and has one not been generated, it will
        not be popped and the parser will go back to conGen stage with nodes
        generated through this.
        A symbol is generated through this method if
        - it is a reserved symbol
        - it has no child and therefore will not be generated through
        promote action
        """
        _, hypo_cache_symbol_index = c.rightmostCache()
        if hypo_cache_symbol_index == NULL_SIDX:
            return None
        gold_cache_symbol_index = c.hyp2gold[hypo_cache_symbol_index]
        for gold_child_idx in gold_graph.headToTail[gold_cache_symbol_index]:
            # if not been generated
            if gold_child_idx not in c.gold2hyp:
                # if not reserved symbols, no generating
                if not gold_graph.isRestrictedIdx(gold_child_idx):
                    return None
                # if is restricted, check if it has no child
                if len(gold_graph.headToTail[gold_child_idx]) > 0:
                    return None
                # otherwise, reserved child with no child
                else:
                    return gold_child_idx

    def generateInSeqSymbol(self, c, gold_graph):
        """A symbol is generated through this method if
        - it is a reserved inseq symbol
        - the entire constituent is inseq or promote reserved symbols so we
          generate the uppermost inseq reserved symbol.
        """
        _, hypo_cache_symbol_index = c.rightmostCache()
        if hypo_cache_symbol_index == NULL_SIDX:
            return None
        gold_cache_symbol_index = c.hyp2gold[hypo_cache_symbol_index]
        for gold_child_idx in gold_graph.headToTail[gold_cache_symbol_index]:
            # if not been generated
            if gold_child_idx not in c.gold2hyp:
                # An inseq index
                if (gold_graph.isInSeqIdx(gold_child_idx) and
                    len(gold_graph.headToTail[gold_child_idx]) == 0):
                    return gold_child_idx
                # A constituent with only reserved symbols.
                # Return the uppermost inseq symbol.
                if gold_graph.isAllReservedConstituent(gold_child_idx):
                    return gold_graph.uppermostInSeqIdx(gold_child_idx)

        # No appropriate child found.
        return None

    # important method subject to change
    def getOracleAction(self, c: CacheConfiguration):
        """Generates the next oracle action from cache configuration, `c`.
        Also, sets the last_action attribute of `c` with the generated action.

        Args:
            c: CacheConfiguration with current parser state.
            gen_method: Method to generate reserved symbols.

        Returns:
            A tuple of the action string, hypothesis index, and gold index.
            The hypothesis and gold indices are None for all actions except
            symID, genID, and symGen.
        """
        gold_graph = c.gold
        widToSymbolID = gold_graph.widToSymbolID
        right_edges = gold_graph.right_edges
        hypo_graph = c.hypothesis
        # Start processing new word.
        # Pop if necessary (no more to symbols in gold graph or rightmost
        # cache element is complete) Otherwise, either generate a new symbol or
        # shift a symbol from the buffer.
        if c.start_word:
            word_idx = c.peekWordIdx()
            #print("word_idx: {}".format(word_idx))
            #print("widToSymbolID: {}".format(widToSymbolID))
            #print("unreserve_gen_method: {}".format(unreserve_gen_method))
            # if somehow a word we don't expect appears
            if word_idx >= 0 and word_idx not in widToSymbolID and self.unreserve_gen_method == UnreserveGenMethod.NONE:
                # NONE gen method will skip a word if it's not in the input alignment.
                # WORD gen method will only skip a word as backup -- see below..
                c.last_action = "emp"
                return "symID:-NULL-", None, None

            hypo_symbol_strings = [con.getValue() for con in c.hypothesis.symbols]
            gold_symbol_strings = [con.getValue() for con in c.gold.symbols]
            if ((len(hypo_symbol_strings) == len(gold_symbol_strings)) and
                    (c.peekWordIdx() >= 0) and
                    (self.unreserve_gen_method == UnreserveGenMethod.WORD)):
                # If we've generated all symbols, but widx_buffer, is not empty, then generate symID:-NULL-.
                c.last_action = "emp"
                return "symID:-NULL-", None, None
            if c.needsPop():
                c.last_action = "POP"
                return "POP", None, None
            next_symbol_idx = hypo_graph.nextSymbolIDX()
            #print("next_symbol_idx: {}".format(next_symbol_idx))

            next_gold_inseq_sidx = self.generateInSeqSymbol(c, gold_graph)
            if next_gold_inseq_sidx is not None:
                unaligned_label = gold_graph.symbolLabel(next_gold_inseq_sidx)
                return "seqGen:" + unaligned_label, next_symbol_idx, next_gold_inseq_sidx

            next_gold_symbol_idx = self.generateReservedSymbol(c, gold_graph)
            if next_gold_symbol_idx is not None:
                unaligned_label = gold_graph.symbolLabel(next_gold_symbol_idx)
                return "symGen:" + unaligned_label, next_symbol_idx, next_gold_symbol_idx

            # Get next unrestricted gold symbol
            next_u_symbol_idx = hypo_graph.nextSymbolIDX(restricted_filtered=True)
            if next_u_symbol_idx not in gold_graph.utof:
                # No more unrestricted symbols to generate.
                return "symID:-NULL-", None, None
            # NB: u_symbol indices should be the same between hypothesis and
            # gold graphs, since these aren't generated
            #print("gold.utof: {}".format(gold_graph.utof))
            #print("hypo.utof: {}".format(hypo_graph.utof))
            next_gold_symbol_idx = gold_graph.utof[next_u_symbol_idx]
            next_u_symbol = gold_graph.symbolLabel(next_gold_symbol_idx)
            next_hypo_symbol_idx = hypo_graph.nextSymbolIDX(restricted_filtered=False)
            #print("next_hypo_symbol_idx: {}".format(next_hypo_symbol_idx))
            #print("next_gold_symbol_idx: {}".format(next_gold_symbol_idx))
            #print("next_u_symbol_idx: {}".format(next_u_symbol_idx))
            #print("next_symbol: {}".format(next_u_symbol))
            #print("word: {}".format(c.wordSeq[word_idx]))

            base, suffix, is_name = split_ulf_atom(next_u_symbol)
            cur_name_token = c.getCurToken()
            cur_sym_token = c.getCurToken(delim="_")
            cur_lemma = c.getCurLemma()
            next_token = c.getNextToken()
            next_lemma = c.getNextLemma()
            #print("in getOracleAction")
            #print("next_u_symbol: {}".format(next_u_symbol))
            #print("split symbol:{}".format(split_ulf_atom(next_u_symbol)))
            #print("cur_name_token: {}".format(cur_name_token))
            #print("cur_sym_token: {}".format(cur_sym_token))
            #print("cur_lemma: {}".format(cur_lemma))
            #print("next_token: {}".format(next_token))
            #print("base: {}".format(base))
            #print("next_gold_symbol_idx: {}".format(next_gold_symbol_idx))
            #print("widToSymbolID: {}".format(widToSymbolID))
            #print("sidIDToWordID: {}".format(gold_graph.symIDToWordID))
            #print("sidToSpan: {}".format(gold_graph.sidToSpan))
            action = None
            # If the symbol is a name and the current string is the base, generate.
            if is_name and cur_name_token != "" and base == cur_name_token:
                action = "nameGen:{}:{}".format(base.replace(" ", "<-SPACE->"), suffix)
            # If the symbol is not a name and the current lemma is the base, generate.
            elif (not is_name) and cur_lemma != "" and base.lower() == cur_lemma.lower():
                action = "lemmaGen:{}:{}".format(base, suffix)
            elif (not is_name) and cur_sym_token != "" and base.lower() == cur_sym_token.lower():
                action = "tokenGen:{}:{}".format(base, suffix)
            # If symbol is a name and the current string + next string is a subset of the base or,
            # the symbol is not a name and the current lemma + the next lemma is a subset of the base,
            # then merge action.
            elif (cur_name_token.strip() is not "") and (next_token.strip() is not "") and \
                    ((is_name and base.startswith(cur_name_token + " " + next_token)) or \
                    (not is_name and base.lower().startswith(cur_lemma.lower() + "_" + next_lemma.lower())) or
                    (not is_name and base.lower().startswith(cur_sym_token.lower() + "_" + next_token.lower()))):
                action = "mergeBuf"
            elif (
                    self.unreserve_gen_method == UnreserveGenMethod.WORD and
                    word_idx >= 0 and
                    (word_idx not in widToSymbolID or
                        widToSymbolID[word_idx] > next_symbol_idx) and
                    (next_symbol_idx in gold_graph.symIDToWordID.keys() and
                        gold_graph.symIDToWordID[next_symbol_idx] > word_idx)
                ):
                # WORD gen method will skip a word if it's not in the input
                # alignment or if we haven't reached the aligned symbol yet
                # (i.e. out-of-order) and the current symbol is waiting on a
                # later word.
                c.last_action = "emp"
                return "symID:-NULL-", None, None
            elif (
                    self.unreserve_gen_method == UnreserveGenMethod.WORD and
                    (next_gold_symbol_idx not in gold_graph.symIDToWordID.keys() or
                    gold_graph.sidToSpan[next_gold_symbol_idx][0] <= word_idx)
                ):
                # If not aligned or the aligned word has already been passed
                # (using naive alignment) just generate it now.
                action = "symGen:" + next_u_symbol
                if action not in self.symGen_actionIDs:
                    action = "symGen:" + UNKNOWN
            elif (self.unreserve_gen_method == UnreserveGenMethod.WORD and word_idx >= 0 and
                    (word_idx not in widToSymbolID or widToSymbolID[word_idx] != next_symbol_idx)):
                # Since symGen didn't handle this, that means the aligned
                # symbol was actually generated already, so skip word.
                c.last_action = "emp"
                return "symID:-NULL-", None, None
            else:
                # Otherwise, just generate the symbol.
                action = "symGen:" + next_u_symbol
                if action not in self.symGen_actionIDs:
                    action = "symGen:" + UNKNOWN

            return action, next_hypo_symbol_idx, next_gold_symbol_idx

        if c.last_action == "ARC" and self.reserve_gen_method == ReserveGenMethod.PROMOTE:
            parent_label, hypo_parent_symbol_idx, gold_parent_symbol_idx = c.needsPromote()
            if parent_label is not None:
                c.last_action = "PROMOTE_SYM"
                return "PROMOTE_SYM:{}".format(parent_label), hypo_parent_symbol_idx, gold_parent_symbol_idx
            else:
                c.last_action = "NOPROMOTE"
                return "NOPROMOTE", None, None

        if "PROMOTE_SYM" == c.last_action:
            c.last_action = "PROMOTE_ARC"
            _, hypo_cache_symbol_idx = c.rightmostCache()
            gold_cache_symbol_idx = c.hyp2gold[hypo_cache_symbol_idx] #gold index
            gold_parent_symbol_idx = gold_graph.tailToHead[gold_cache_symbol_idx]
            gold_parent_symbol = gold_graph.getSymbol(gold_parent_symbol_idx)
            hypo_parent_symbol_idx = c.hypothesis.nextSymbolIDX()
            hypo_graph.headToTail[hypo_parent_symbol_idx].add(hypo_cache_symbol_idx)
            arc_label = gold_parent_symbol.getRelStr(gold_cache_symbol_idx)
            return "PROMOTE_ARC:" + arc_label, None, None

        # ARC decision and connect.
        if c.last_action == "PUSH" or c.last_action == "POP" or c.last_action == "PROMOTE_ARC":
            num_connect = c.cache_size - 1
            _, hypo_cand_symbol_idx = c.getCache(num_connect)
            gold_cand_symbol_idx = c.hyp2gold[hypo_cand_symbol_idx]
            arcs = []
            c.last_action = "ARC"
            # for each element in the cache, decide what to do
            for cache_idx in range(num_connect):
                cache_word_idx, hypo_cache_symbol_idx = c.getCache(cache_idx)

                # Current cache value is a placeholder.
                if hypo_cache_symbol_idx == NULL_SIDX:
                    arcs.append("O")
                    continue

                gold_cache_symbol_idx = c.hyp2gold[hypo_cache_symbol_idx]

                # Get edges where the candidate symbol is the parent.
                if gold_cache_symbol_idx in gold_graph.headToTail[gold_cand_symbol_idx]:
                    # Enforce bottom-up. Don't allow if child is not fully constructed.
                    if len(hypo_graph.headToTail[hypo_cache_symbol_idx]) < len(gold_graph.headToTail[gold_cache_symbol_idx]):
                        arcs.append("O")
                    # Don't add edge if it already exists in the hypothesis.
                    elif hypo_cache_symbol_idx in hypo_graph.symbols[hypo_cand_symbol_idx].tail_ids:
                        arcs.append("O")
                    else:
                        gold_cand_symbol = gold_graph.getSymbol(gold_cand_symbol_idx)
                        arc_label = "L-" + gold_cand_symbol.getRelStr(gold_cache_symbol_idx)
                        if arc_label not in self.arc_actionIDs:
                            arc_label = "L-" + UNKNOWN
                        arcs.append(arc_label)
                        hypo_graph.headToTail[hypo_cand_symbol_idx].add(hypo_cache_symbol_idx)

                # Get edges where the candidate symbol is the child.
                elif gold_cand_symbol_idx in gold_graph.headToTail[gold_cache_symbol_idx]:
                    # Enforce bottom-up. Don't allow if child is not fully constructed.
                    if len(hypo_graph.headToTail[hypo_cand_symbol_idx]) < len(gold_graph.headToTail[gold_cand_symbol_idx]):
                        arcs.append("O")
                    # Don't add edge if it already exists in the hypothesis.
                    elif hypo_cand_symbol_idx in hypo_graph.symbols[hypo_cache_symbol_idx].tail_ids:
                        arcs.append("O")
                    else:
                        gold_cache_symbol = gold_graph.getSymbol(gold_cache_symbol_idx)
                        arc_label = "R-" + gold_cache_symbol.getRelStr(gold_cand_symbol_idx)
                        if arc_label not in self.arc_actionIDs:
                            arc_label = "R-" + UNKNOWN
                        arcs.append(arc_label)
                        hypo_graph.headToTail[hypo_cache_symbol_idx].add(hypo_cand_symbol_idx)

                else:
                    arcs.append("O")

            return "ARC:" + "#".join(arcs), None, None
        if c.last_action in ["symID", "symGen", "seqGen", "nameGen", "lemmaGen", "tokenGen"]:
            c.last_action = "PUSH"
            #print()
            #print("BERFORE CHOOSE VERTEX")
            #print("GOLD graph")
            #print(c.gold)
            #print("HYPOTHESIS graph")
            #print(c.hypothesis)
            #print("Stack")
            #print(c.stack)
            #print("Cache")
            #print(c.cache)
            cache_idx = self.chooseVertex(c, right_edges)
            return "PUSH:%d" % cache_idx, None, None
        print("Unable to proceed from last action:" + c.last_action, sys.stderr)
        print("Configuration: {}".format(c))
        print("gold graph: {}".format(gold_graph))
        print("hypo graph: {}".format(hypo_graph))


import sys
import numpy as np
import copy
from typing import Dict
from collections import defaultdict
from postprocessing.amr_format import get_amr
from oracle.cacheConfiguration import CacheConfiguration
from oracle.utils import FeatureType, ReserveGenMethod, UnreserveGenMethod
from lisp.sanity_checker import ulf_sanity_checker
from exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()

def sort_hypotheses(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)

def generateAMR(hypo, trans_system, sent_anno):
    """Generates an postprocessing.amr_graph.AMR instance from a decoder
    Hypothesis state, a transition system, and an annotated sentence.
    """
    symbol_line_reprs = hypo.trans_state.toConll(
            unreserve_gen_method=trans_system.unreserve_gen_method)
    category_map = sent_anno.map_info
    # Fill in "NONE" categories for generated symbols.
    new_category_map = [
            category_map[hypo.trans_state.next_symbol_data.all2orig[i]]
                if i in hypo.trans_state.next_symbol_data.all2orig
                else "NONE||{}".format(val[0])
            for i, val in enumerate(symbol_line_reprs)
    ]
    return get_amr(symbol_line_reprs, new_category_map)


class Hypothesis(object):
    def __init__(self,
            actions,
            log_ps,
            state_embeddings,
            cache_config=None,
            sanity_check_arcs=False,
            s_memory_bank=None,
        ):
        self.actions = actions # store all actions
        self.log_probs = log_ps # store log_probs for each time-step
        # Arbitrary state for tracking parser embedding states.
        self.state_embeddings = state_embeddings
        self.trans_state = cache_config
        self.cache_idx = 0
        self.action_repeat_counts = defaultdict(int)
        self.sanity_check_arcs = sanity_check_arcs

        # all the propagating neural embedding states
        self.s_memory_bank = s_memory_bank
        #self.w_memory = []
        #self.s_memory = []

        # Tracking for debugging.
        self.foci = []

    def addAction(self, action):
        self.actions.append(action)

    def actionSeqStr(self, action_vocab_fn):
        return "#".join([action_vocab_fn(action_id) for action_id in self.actions])

    def actionStrs(self, action_vocab_fn):
        return [action_vocab_fn(aid) for aid in self.actions]

    def extend(self,
            system,
            action_id,
            log_prob,
            state_embeddings,
            action_vocab_fn,
            s_memory_bank=None,
            gold=False,
            print_info=False,
            sent_anno=None):
        """Generates a new hypothesis state based on applying the given action
        (with appropriate context) to the current hypothesis. This new
        hypothesis is returned. Returns None if the action cannot be applied.
        If gold is set to True, then the action applicability tests are ignored,
        simply applying it anyway. The assumption is that this is generated from
        the oracle so it must be applicable.
        """
        action = action_vocab_fn(action_id)
        if (
                (not gold) and
                (not system.canApply(self.trans_state, action,
                     use_refined=False, cache_idx=self.cache_idx,
                     pop_refined=True, # enforce connectedness
                     no_trailing_seqgen=True, # stop trailing symbols
                     ))
            ):
            return None

        cache_size = self.trans_state.cache_size
        new_config = CacheConfiguration(
            cache_size,
            len(self.trans_state.widx_buffer),
            self.trans_state, # Initialize from another config
        )
        new_foci = self.foci + [system.focus_manager.get(new_config)]
        next_cache_idx = self.cache_idx
        if print_info:
            old_word_focus, old_symbol_focus = system.focus_manager.get(self.trans_state)
            new_word_focus, new_symbol_focus = system.focus_manager.get(new_config)
            print("In Hypothesis.extend()")
            print("new_config.phase: {}".format(new_config.phase))
            print("new word focus: {}".format(new_word_focus))
            print("new symbol focus: {}".format(new_symbol_focus))
            print("old word focus: {}".format(old_word_focus))
            print("old symbol focus: {}".format(old_symbol_focus))
            print("new symbol data: {}".format(new_config.next_symbol_data))
            print("old symbol data: {}".format(self.trans_state.next_symbol_data))
            print("nextSymbolIDX: {}".format(new_config.hypothesis.nextSymbolIDX()))
            print("action: {}".format(action))
            print("symbols: {}".format([con.getValue() for con in new_config.hypothesis.symbols]))
            print("widx_width: {}".format(new_config.widx_width))
            print("widxBufferSize: {}".format(new_config.widxBufferSize()))
            print("widx_buffer: {}".format(new_config.widx_buffer))
        new_actions = self.actions + [action_id]
        new_probs = self.log_probs + [log_prob]
        new_hyp = Hypothesis(
            new_actions,
            new_probs,
            state_embeddings,
            new_config,
            self.sanity_check_arcs,
            s_memory_bank,
        )
        new_hyp.foci = new_foci
        new_hyp.action_repeat_counts = copy.deepcopy(self.action_repeat_counts)
        if new_config.phase == FeatureType.SHIFTPOP:
            if action == "SHIFT": # Process the next symbol.
                assert self.trans_state.next_symbol_data.all == self.trans_state.hypothesis.nextSymbolIDX(), \
                        "all_symbol_focus: {}\tnextsymbolIDX(): {}".format(
                            self.trans_state.next_symbol_data.all,
                            self.trans_state.hypothesis.nextSymbolIDX(),
                        )
                next_symbol_focus = system.focus_manager.extract_symbol_idx(
                        self.trans_state.next_symbol_data, self.trans_state.phase, True)
                curr_symbol = new_config.getSymbol(next_symbol_focus, orig=True)
                oracle_action = "symID:" + curr_symbol
                if new_config.isUnalign(next_symbol_focus):
                    oracle_action = "symGen:" + curr_symbol
                next_cache_idx = cache_size - 2
                system.apply(new_config, oracle_action)
            elif action == "POP":
                system.apply(new_config, action)
            elif action == "SEQGEN": # Generate next symbol next time.
                new_config.phase = FeatureType.SYMSELECT
            elif action == "WORDGEN": # Generate next symbol from a word.
                new_config.phase = FeatureType.WORDGEN
            elif action == "MERGEBUF": # Merge next two elements.
                system.apply(new_config, "mergeBuf")
            elif action == "symID:-NULL-":
                system.apply(new_config, action)
            elif action == "GENSYM": # Generate the symbol at the next step.
                new_config.phase = FeatureType.SYMSELECT
            else:
                assert False, "UNKNOWN SHIFTPOP action, {}".format(action)
        elif new_config.phase == FeatureType.SYMSELECT:
            assert "seqGen" in action or "symGen" in action, "action: {}".format(action)
            system.apply(new_config, action)
        elif new_config.phase == FeatureType.WORDGEN:
            if action == "LEMMA":
                new_config.phase = FeatureType.WORDGEN_LEMMA
            elif action == "NAME":
                new_config.phase = FeatureType.WORDGEN_NAME
            elif action == "TOKEN":
                new_config.phase = FeatureType.WORDGEN_TOKEN
            else:
                assert False, "UNKNOWN WORDGEN action, {}".format(action)
        elif new_config.phase in [
                FeatureType.WORDGEN_LEMMA,
                FeatureType.WORDGEN_NAME,
                FeatureType.WORDGEN_TOKEN
                ]:
            assert "SUFFIX" in action, "action: {}".format(action)
            suffix = action[7:]
            oracle_action = None
            if new_config.phase == FeatureType.WORDGEN_LEMMA:
                oracle_action = "lemmaGen:{}:{}".format(new_config.getCurLemma(), suffix)
            elif new_config.phase == FeatureType.WORDGEN_TOKEN:
                oracle_action = "tokenGen:{}:{}".format(new_config.getCurToken(delim="_"), suffix)
            else:
                oracle_action = "nameGen:{}:{}".format(new_config.getCurToken(), suffix)
            system.apply(new_config, oracle_action)

        elif new_config.phase == FeatureType.PROMOTE:
            assert action in ["PROMOTE", "NOPROMOTE"], action
            if action == "PROMOTE":
                new_config.phase = FeatureType.PROMOTE_SYM
                new_hyp.action_repeat_counts['promote'] += 1
            if action == "NOPROMOTE":
                new_config.phase = FeatureType.SHIFTPOP
                new_hyp.action_repeat_counts['promote'] = 0
        elif new_config.phase == FeatureType.PROMOTE_SYM:
            assert action.startswith("PROMOTE_SYM:"), action
            system.apply(new_config, action)
        elif new_config.phase == FeatureType.PROMOTE_ARC:
            assert action.startswith("PROMOTE_ARC:"), action
            system.apply(new_config, action)

        elif new_config.phase == FeatureType.PUSHIDX:
            prev_action = action_vocab_fn(self.actions[-1])
            # Can only push after shift or symbol generation.
            assert (prev_action == "SHIFT") or ("seqGen" in prev_action) or \
                    ("SUFFIX" in prev_action), "prev_action: {}".format(prev_action)
            assert next_cache_idx == cache_size - 2, \
                    "next_cache_idx: {}\t(cache_size - 2): {}".format(
                        next_cache_idx,
                        cache_size - 2,
                    )
            system.apply(new_config, action)
        # Judge if there is an arc
        elif new_config.phase == FeatureType.ARCBINARY:
            if action == "NOARC": # No arc made to current cache index
                if next_cache_idx == 0: # Already the last cache index
                    # Next stage should shift or pop
                    next_cache_idx = cache_size - 2
                    if system.reserve_gen_method == ReserveGenMethod.PROMOTE:
                        new_config.phase = FeatureType.PROMOTE
                    else:
                        new_config.phase = FeatureType.SHIFTPOP
                else: # Check next cache index
                    next_cache_idx -= 1
            else: # Then process the label
                assert action == "ARC", "action: {}".format(action)
                new_config.phase = FeatureType.ARCCONNECT
        else:
            assert new_config.phase == FeatureType.ARCCONNECT, \
                    "new_config.phase: {}".format(new_config.phase)
            # Perform arc connection
            oracle_action = "ARC%d:%s" % (next_cache_idx, action)
            system.apply(new_config, oracle_action)
            if next_cache_idx == 0:
                next_cache_idx = cache_size - 2
                if system.reserve_gen_method == ReserveGenMethod.PROMOTE:
                    assert new_config.phase == FeatureType.PROMOTE, \
                            "new_config.phase: {}".format(new_config.phase)
                else:
                    assert new_config.phase == FeatureType.SHIFTPOP, \
                            "new_config.phase: {}".format(new_config.phase)
            else:
                next_cache_idx -= 1

        if (action == "SEQGEN") or (action == "GENSYM"):
            new_hyp.action_repeat_counts['seqgen'] += 1
        elif (action == "WORDGEN") or ("symID" in action):
            new_hyp.action_repeat_counts['seqgen'] = 0
        else:
            new_hyp.action_repeat_counts['seqgen'] += \
                    self.action_repeat_counts['seqgen']
        new_hyp.cache_idx = next_cache_idx
        if self.sanity_check_arcs and self.trans_state.phase == FeatureType.ARCCONNECT:
            # Check that after making this arc, the current hypothesis still
            # satisfies the sanity checker.
            ulfamr_str = generateAMR(new_hyp, system, sent_anno).to_amr_string()
            try:
                if ulf_sanity_checker.exists_bad_pattern(ulfamr_str):
                    return None
            except Exception as e:
                return None

        return new_hyp

    def readOffUnalignWords(self):
        symbol_align = self.trans_state.symbolAlign

        # If all symbols are read, should also move word pointer to the last.
        if self.trans_state.next_symbol_data.orig >= len(symbol_align):
            self.trans_state.clearWidxBuffer()
            self.trans_state.next_word_idx = len(self.trans_state.wordSeq)
            return

        length = len(self.trans_state.wordSeq)
        while (self.trans_state.next_word_idx not in self.trans_state.widTosid) and self.trans_state.next_word_idx < length: # Some words are unaligned.
            popped = self.trans_state.popWidxBuffer()
            assert popped == self.trans_state.next_word_idx, \
                    "popped: {}\tself.trans_state.next_word_idx: {}".format(popped, self.trans_state.next_word_idx)
            self.trans_state.next_word_idx += 1

    def extractFeatures(self, uniform=False):
        # At first step, decide whether to shift or pop.
        word_idx, symbol_idx, cache_idx = (
                self.trans_state.next_word_idx,
                self.trans_state.next_symbol_data.all,
                self.cache_idx,
        )
        if (self.trans_state.phase == FeatureType.ARCBINARY or
                self.trans_state.phase == FeatureType.ARCCONNECT):
            assert self.actions, "Empty action sequence start without shift or pop"
            assert self.cache_idx != -1, "Cache related operation without cache index."
            word_idx, symbol_idx = self.trans_state.rightmostCache()
        if (self.trans_state.phase == FeatureType.PROMOTE or
                self.trans_state.phase == FeatureType.PROMOTE_SYM):
            word_idx, symbol_idx = self.trans_state.rightmostCache()
            cache_idx = 0
        if self.trans_state.phase == FeatureType.PROMOTE_ARC:
            cache_idx = self.trans_state.cache_size - 1
        return self.trans_state.extractFeatures(
                self.trans_state.phase,
                word_idx,
                symbol_idx,
                self.cache_idx,
                uniform_arc=uniform,
        )

    def latest_action(self):
        return self.actions[-1]

    def avg_log_prob(self):
        return np.sum(self.log_probs[1:]) / (len(self.actions) - 1)

    def probs2string(self):
        out_string = ""
        for prob in self.log_probs:
            out_string += " %.4f" % prob
        return out_string.strip()

    def check_repeat_limits(self, max_repeats: Dict[str, int]) -> bool:
        """Checks whether the hypothesis is in the action repeat limits.
        """
        for cat, limit in max_repeats.items():
            if (cat in self.action_repeat_counts and
                    self.action_repeat_counts[cat] >= limit):
                return False
        return True

    def amr_string(self):
        pass



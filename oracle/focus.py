from typing import Dict, List, Tuple, Type, Union
import copy
from .utils import (
    FeatureType,
    FocusMethod,
    ReserveGenMethod,
    TypeCheckingMethod,
    UnreserveGenMethod,
)

"""

Here are some notes on the relationship between the FocusManager.extract_symbol_idx value, NextSymbolData and the indices in the hypothesis ULFAMRGraph.

The basic indexing for symbols originates in the ULFAMRGraph internal lists and
it has associated indexing for the next buffer-original symbol only and
symbol indexing including those generated in the parsing process.

NextSymbolData takes the information in ULFAMRGraph and stores all the indexing
relationships between original indexing and all-indexing. It also includes
the same counting indexing as ULFAMRGraph that can be accessed more lightly.

FocusManager.extract_symbol_idx takes NextSymbolData and generates the
actual indexing that is desired based on two factors:
    1. the parsing phase
        For some versions of the FocusManager, the phase will determine
        whether the original or all indexing is used.
    2. the indexing offset of symbols
        The focus manager is used for indexing into the neural network
        parser, so this allows the indexing to be offset between the
        neural network symbols and the ulfamr graph. This is designed to
        allow space for special tokens (e.g. start of sequence, unk)
        at the start of the neural network word buffer sequnce.

Because of this, what you'll see in the code is that code that exclusively
manages the transition system accesses the NextSymbolData values directly,
whereas code that's meant as inputs to the neural network uses the FocusManager
interface. Also, the features are extracted directly from the AMR graphs, so
these will also access NextSymbolData directly.
"""

class NextSymbolData:
    """Generic next symbol tracking data class.

    This class must subsume all the attributes needed for every type of symbol
    tracker since the CacheConfiguration class is designed to be agnostic to
    the transition system configuration, simply modeling the current state of
    the stack, cache, and buffer.
    """
    orig: int = 0
    # Monotonically increasing focus over all symbols in the transition
    # system. When a new symbol is generated or shifted, that becomes the
    # new focus.
    all: int = 0
    all2orig: Dict[int,int] = {}
    orig2all: Dict[int,int] = {}

    def __repr__(self):
        keyvals = ", ".join([
            str(key) + ":" + str(value) for
            key, value in vars(self).iteritems()
        ])
        return "NextSymbolData({})".format(keyvals)


class NextSymbolDataUpdater:
    """A generic class for updating of the next symbol index data after various
    actions, as well as mapping between different types of symbol indexings.
    """
    @staticmethod
    def update(data: NextSymbolData, action: str) -> NextSymbolData:
        """Returns an updated next symbol index data instance based on the
        given action without modifying the original.
        """
        raise NotImplementedError


class WordPromoteSymbolDataUpdater(NextSymbolDataUpdater):
    """A symbol tracker for a transition system with word buffers and
    promote-based unaligned symbol generation
    """
    @staticmethod
    def update(data: NextSymbolData, action: str) -> NextSymbolData:
        """Takes an oracle action string and updates the foci accordingly.
        This must be the oracle abbreviated actions, not the simplified
        generated ones.
            So rather than "WORDGEN", "LEMMA", then "SUFFIX:PRO"
            just supply "lemmaGen:HE:PRO".
        Basically, the action argument here must match what would be supplied
        to cacheTransition.apply(
        """
        newdata = copy.deepcopy(data)
        if action.startswith("seqGen") or action.startswith("symGen"):
            newdata.all += 1
        elif action.startswith("lemmaGen") or action.startswith("tokenGen") or action.startswith("nameGen"):
            newdata.all += 1
        elif "PROMOTE_SYM:" in action:
            newdata.all += 1
        elif not (action.startswith("PUSH:") or action.startswith("ARC") or
                action.startswith("PROMOTE_ARC:") or
                action in ["PROMOTE", "NOPROMOTE", "NOARC", "POP", "symID:-NULL-", "mergeBuf"]):
            errmsg = "Unknown action for WordPromoteSymbolTracker: {}".format(action)
            raise ValueError(errmsg)
        return newdata


class SymbolBufferSymbolDataUpdater(NextSymbolDataUpdater):
    """A symbol tracker for a transition system with symbol buffers.
    This allows seqGen, symGen, and PROMOTE_SYM for generating symbols that
    don't originate in the buffer.
    """
    @staticmethod
    def update(data: NextSymbolData, action: str) -> NextSymbolData:
        newdata = copy.deepcopy(data)
        if action == "SHIFT":
            newdata.all2orig[newdata.all] = newdata.orig
            newdata.orig2all[newdata.orig] = newdata.all
            newdata.orig += 1
            newdata.all += 1
        elif action.startswith("seqGen") or action.startswith("symGen"):
            newdata.all += 1
        elif "PROMOTE_SYM:" in action:
            newdata.all += 1
        elif not (action.startswith("PUSH:") or action.startswith("ARC") or
                action.startswith("PROMOTE_ARC:") or
                action in ["PROMOTE", "NOPROMOTE", "NOARC", "POP", "symID:-NULL-", "mergeBuf"]):
            errmsg = "Unknown action for WordPromoteSymbolTracker: {}".format(action)
            raise ValueError(errmsg)
        return newdata


class FocusManager:
    """A generic central class for keeping track of all next symbol index and
    generating word/symbol focuses during specific phases. Word index tracking
    is done in the cache configuration because it is directly tied to the input
    buffer.
    """
    @classmethod
    def update_symbol_data(cls, data: NextSymbolData,
            action: str) -> NextSymbolData:
        """Updates the next symbol index tracker based on the given action.
        """
        return cls.SYMBOL_DATA_UPDATER.update(data, action)

    def extract_symbol_idx(self, data: NextSymbolData, phase: FeatureType,
            ignore_offset: bool) -> int:
        """Extracts the symbol index from 'data'.
        """
        raise NotImplementedError

    def extract_word_idx(self, raw_idx: int, c: 'CacheConfiguration') -> int:
        """Extracts the word index from the cache configuration.
        """
        if c.phase == FeatureType.PUSHIDX:
            raw_idx = c.cand_vertex.word_idx
        if raw_idx < 0:
            # Assume negative values are negations of the appended order that
            # special tokens are appended to the end.
            return len(c.wordSeq) + abs(raw_idx) - 1
        else:
            return raw_idx

    def get(self, c: 'CacheConfiguration'
            ) -> Tuple[Union[int, List[int]], Union[int, List[int]]]:
        """Returns the a two element tuple containing the word focus and the
        symbol focus. Each may be a list, for managers that manage multiple
        foci.
        """
        raise NotImplementedError

    @staticmethod
    def from_transition_system(ts: 'CacheTransition') -> 'FocusManager':
        generation_method_triple = (
                ts.reserve_gen_method,
                ts.unreserve_gen_method,
                ts.focus_method,
        )
        manager = None
        if (generation_method_triple ==
                (ReserveGenMethod.PROMOTE, UnreserveGenMethod.WORD, FocusMethod.NONE)):
            manager = WordPromoteNextOnlyFocusManager()
        elif (generation_method_triple ==
                (ReserveGenMethod.PROMOTE, UnreserveGenMethod.WORD, FocusMethod.CACHE)):
            manager = WordPromoteCacheArcFocusManager()
        else:
            errmsg = ("A focus manager for the supplied transition system "
                      "configuration is not available: {}".format(generation_method_triple))
            raise ValueError(errmsg)
        manager.buffer_offset = ts.buffer_offset
        return manager


class WordPromoteNextOnlyFocusManager(FocusManager):
    """A simple single access focus manager for a transition system with a word
    buffer and promotion-based unaligned symbol generation.

    This manager always returns the next word and symbol indices as focuses.
    """
    SYMBOL_DATA_UPDATER = WordPromoteSymbolDataUpdater

    def extract_symbol_idx(self, data: NextSymbolData, phase: FeatureType,
            ignore_offset: bool=False) -> int:
        if ignore_offset:
            return data.all
        else:
            return data.all + self.buffer_offset - 1

    def get(self, c: 'CacheConfiguration') -> Tuple[int, int]:
        """Gets a tuple of the the focuses based on the cache configuration
        state.
        """
        return (
            self.extract_word_idx(c.next_word_idx, c),
            self.extract_symbol_idx(c.next_symbol_data, c.phase),
        )


class WordPromoteCacheArcFocusManager(FocusManager):
    """A single access focus manager for a transition system with a word buffer
    and promotion-based unaligned symbol generation, which returns the rightmost
    cache index for the word and symbol focus when making arcs (and otherwise
    appropriate). Otherwise, it returns the tracker values.
    """
    SYMBOL_DATA_UPDATER = WordPromoteSymbolDataUpdater
    CACHE_PHASES = [
            FeatureType.ARCBINARY,
            FeatureType.ARCCONNECT,
            FeatureType.PROMOTE,
            FeatureType.PROMOTE_ARC,
            FeatureType.PROMOTE_SYM,
    ]

    def extract_symbol_idx(self, data: NextSymbolData, phase: FeatureType,
            ignore_offset: bool=False) -> int:
        if ignore_offset:
            return data.all
        else:
            return data.all + self.buffer_offset - 1 # assume one symbol is the start symbol

    def get(self, c: 'CacheConfiguration') -> Tuple[int, int]:
        """Gets a tuple of the the focuses based on the cache configuration
        state.
        """
        if c.phase in self.CACHE_PHASES:
            raw_widx, raw_sidx = c.rightmostCache()
            return (
                self.extract_word_idx(raw_widx, c),
                raw_sidx + self.buffer_offset,
            )
        else:
            return (
                self.extract_word_idx(c.next_word_idx, c),
                self.extract_symbol_idx(c.next_symbol_data, c.phase),
            )


import argparse
import sys
import time
from collections import defaultdict
from .ioutil import *
from .oracle_data import *
from .cacheTransition import CacheTransition
from .cacheConfiguration import CacheConfiguration
from .focus import FocusManager
from .utils import (
    FocusMethod, ReserveGenMethod, UnreserveGenMethod,
    FeatureType,
    loadTokens,
    NONE_WIDX, NULL_WIDX, NULL_SIDX,
)
from .exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()

NULL = "-NULL-"
UNKNOWN = "-UNK-"
class CacheTransitionParser(object):
    def __init__(self,
            size,
            inseq_symbol_file=None,
            promote_symbol_file=None,
            reserve_gen_method=ReserveGenMethod.NONE,
            unreserve_gen_method=UnreserveGenMethod.NONE,
            focus_method=FocusMethod.NONE,
            buffer_offset=0,
            ):
        self.cache_size = size
        self.connectWordDistToCount = defaultdict(int)
        self.nonConnectDistToCount = defaultdict(int)
        self.depConnectDistToCount = defaultdict(int)
        self.depNonConnectDistToCount = defaultdict(int)

        self.wordSymbolCounts = defaultdict(int)
        self.lemmaSymbolCounts = defaultdict(int)
        self.mleSymbolID = defaultdict(int)
        self.lemMLESymbolID = defaultdict(int)
        self.symbolIDDict = defaultdict(int)
        self.unalignedSet = defaultdict(int)

        self.wordIDs = {}
        self.lemmaIDs = {}
        self.symbolIDs = {}
        self.posIDs = {}
        self.nerIDs = {}
        self.depIDs = {}
        self.arcIDs = {}

        self.buffer_offset = buffer_offset

        if inseq_symbol_file is not None and inseq_symbol_file != "":
            self.inseq_syms = set(open(inseq_symbol_file, 'r').read().strip().split(' '))
        else:
            self.inseq_syms = set()
        if promote_symbol_file is not None and promote_symbol_file != "":
            self.promote_syms = set(open(promote_symbol_file, 'r').read().strip().split(' '))
        else:
            self.promote_syms = set()
        self.rsyms = self.inseq_syms | self.promote_syms
        self.reserve_gen_method = reserve_gen_method
        self.unreserve_gen_method = unreserve_gen_method
        self.focus_method = focus_method

    def OracleExtraction(
            self,
            data_dir,
            output_dir,
            decode=False,
            uniform=False,
            for_ulf=False,
            predicted_symbols=False,
            pred_symbol_path="",
        ):
        """Extracts the oracle with this parser using the given configurations.

        Args:
            data_dir: directory with basic input data (token, lemma, pos, ner, alignments, etc.)
            output_dir: directory to write the oracle output
            decode: whether this oracle is for decoding (vs. training)
            uniform: whether to use uniform arc features (gene: not sure what this means)
            for_ulf: whether this oracle is for ULF parsing
            predicted_symbols: whether to use predicted (vs. gold parse) symbols
            pred_symbol_path: file path to predicted symbols
        """
        # reason about.
        #print("Input data directory:", data_dir)
        #print("Output directory:", output_dir)
        os.system("mkdir -p %s" % output_dir)

        oracle_output = os.path.join(output_dir, "oracle_output.txt")
        if decode:
            json_output = os.path.join(output_dir, "oracle_decode.json")
        else:
            json_output = os.path.join(output_dir, "oracle_examples.json")

        dataset = loadDataset(data_dir, ulfdep=for_ulf)
        if predicted_symbols:
            pred_symbols = loadTokens(delim="\t")
            assert len(pred_symbols) == dataset.dataSize(), \
                    "pred_symbols: {}\tdataset: {}".format(len(pred_symbols), dataset.dataSize())
        oracle_set = OracleData()   # Empty Oracle Data

        max_oracle_length = 0.0
        total_oracle_length = 0.0

        data_num = dataset.dataSize()
        # A void method that generates lists and dictionaries to get tokens in sorted order and
        # with a dictionary that takes token in and return its index
        dataset.genDictionaries()

        cache_transition = CacheTransition(
            self.cache_size,
            dataset.known_symbols,
            dataset.known_rels,
            inseq_symbols=self.inseq_syms,
            promote_symbols=self.promote_syms,
            reserve_gen_method=self.reserve_gen_method,
            unreserve_gen_method=self.unreserve_gen_method,
            focus_method=self.focus_method,
            buffer_offset=self.buffer_offset,
        )

        cache_transition.makeTransitions()

        push_actions = defaultdict(int)
        arc_binary_actions = defaultdict(int)
        arc_label_actions = defaultdict(int)
        gensym_actions = defaultdict(int)
        seqgen_actions = defaultdict(int)
        lemmagen_actions = defaultdict(int)
        namegen_actions = defaultdict(int)
        tokengen_actions = defaultdict(int)
        promote_binary_actions = defaultdict(int)
        promote_sym_actions = defaultdict(int)
        promote_arc_actions = defaultdict(int)
        num_pop_actions = 0
        num_shift_actions = 0
        num_gensym_actions = 0
        num_seqgen_actions = 0
        num_lemmagen_actions = 0
        num_tokengen_actions = 0
        num_namegen_actions = 0
        num_mergebuf_actions = 0
        num_promote_actions = 0
        feat_dim = -1
        success_num = 0.0

        # Run oracle on the training data.
        with open(oracle_output, 'w') as oracle_wf:
            for sent_idx in range(data_num):
                inst = dataset.getInstance(sent_idx)  # ioutil.DatasetInstance

                # ULFAMR graph setup
                ulfamr_graph = inst.ulfamr
                symbol_seq = ulfamr_graph.getSymbolSeq()
                map_info_seq = ulfamr_graph.getMapInfoSeq()
                ulfamr_graph.initTokens(inst.tok)
                ulfamr_graph.initLemma(inst.lem)
                ulfamr_graph.buildRestrictedSymbolMap(self.rsyms, self.inseq_syms, self.promote_syms)
                ulfamr_graph.fillGeneratingChildren()
                ulfamr_graph.buildUnrestrictedDescendantMap()

                # This should be the length of the word sequence.
                length = len(inst.tok)
                #length = len(symbol_seq)
                #print("Sentence: %s" %" ".join(inst.tok))
                #print("Symbol Sentence: %s" %" ".join(symbol_seq))

                # Set up cache configuration
                c = CacheConfiguration(self.cache_size, length, oracle=True)
                c.wordSeq, c.lemSeq, c.posSeq, c.nerSeq = inst.tok, inst.lem, inst.pos, inst.ner
                c.tree = inst.dep
                c.symbolSeq = (
                        symbol_seq
                        if self.reserve_gen_method == ReserveGenMethod.NONE
                        else ulfamr_graph.getRawSymbolSeq()
                )
                c.buildSymbolToTypes()
                c.setGold(
                    ulfamr_graph,
                    update_widx=(self.unreserve_gen_method != UnreserveGenMethod.WORD),
                )

                # Partial computation
                feat_seq = []
                focus_align = []
                start_time = time.time()
                oracle_seq = []
                succeed = True
                symbol_to_word = []
                #print("sent_idx: {}\nwidToSymbolID: {}".format(sent_idx, ulfamr_graph.widToSymbolID))
                #print(inst.tok)
                #print(symbol_seq)
                #print(ulfamr_graph.sidToSpan)
                #print("wordSeq: {}".format(c.wordSeq))
                #print("lemSeq: {}".format(c.lemSeq))

                #print("start_time: {}".format(start_time))
                # Search Process
                while not cache_transition.isTerminal(c):
                    #word_focus, symbol_focus = cache_transition.focus_manager.get(c)
                    #print("")
                    #print("Oracle symbol focus: {}".format(symbol_focus))
                    #print("Configuration: {}".format(c))
                    #print("Hypothesis symbols: {}".format([sym.getValue() for sym in c.hypothesis.symbols]))
                    #print("stack size: {}".format(c.stackSize()))
                    #print("word idx buffer size: {}".format(c.widxBufferSize()))
                    #print("word idx buffer: {}".format(c.widx_buffer))
                    #print("widx_width: {}".format(c.widx_width))
                    #print("hypothesis counter: {}".format(c.hypothesis.counter))
                    #print("len symbolSeq: {}".format(len(c.symbolSeq)))
                    #print("symbolSeq: {}".format(c.symbolSeq))
                    #print("wordSeq: {}".format(c.wordSeq))
                    #print("lemSeq: {}".format(c.lemSeq))
                    #print("len gold symbols: {}".format(len(c.gold.symbols)))
                    #print("gold symbols: {}".format([sym.getValue() for sym in c.gold.symbols]))
                    #print("cache state (before oracle action): %s" % c)
                    #print("cache state (before oracle action): %s" % [c.hypothesis.symbolLabel(cache_val.symbol_idx) for cache_val in c.cache])
                    #print("cur_time: {}".format(time.time()))
                    oracle_action, hypo_idx, gold_idx = \
                            cache_transition.getOracleAction(c)
                    #print("Oracle action: {}".format(oracle_action))
                    #print("Tree prior to apply %s: \n %s" %(oracle_action, c.hypothesis))
                    #print("Print Current Cache:\n %s" %c.cache)
                    #print("Gold: \n %s" %(c.gold))
                    if time.time() - start_time > 8000.0:
                        print("Overtime sentence #%d" % sent_idx, file=sys.stderr)
                        print("Sentence: %s" % " ".join(inst.tok), file=sys.stderr)
                        succeed = False
                        break
                    #print("Sentence: {}".format(" ".join(inst.tok)))

                    if "NOPROMOTE" == oracle_action:
                        # Use cache focus.
                        cache_word_idx, cache_symbol_idx = c.rightmostCache()
                        feat_seq.append(c.extractFeatures(FeatureType.PROMOTE,
                            cache_word_idx, cache_symbol_idx, 0, uniform_arc=uniform))
                        oracle_seq.append("NOPROMOTE")
                        focus_align.append(cache_transition.focus_manager.get(c))
                        cache_transition.apply(c, oracle_action)
                        promote_binary_actions['N'] += 1

                    elif "PROMOTE_SYM:" in oracle_action:
                        _, symbol = oracle_action.split(":")
                        # Use cache focus.
                        cache_word_idx, cache_symbol_idx = c.rightmostCache()
                        # First add PROMOTE action.
                        feat_seq.append(c.extractFeatures(FeatureType.PROMOTE,
                            cache_word_idx, cache_symbol_idx, 0, uniform_arc=uniform))
                        oracle_seq.append("PROMOTE")
                        focus_align.append(cache_transition.focus_manager.get(c))
                        promote_binary_actions['Y'] += 1
                        # Choose the symbol.
                        feat_seq.append(c.extractFeatures(FeatureType.PROMOTE_SYM,
                            cache_word_idx, cache_symbol_idx, 0, uniform_arc=uniform))
                        oracle_seq.append(oracle_action)
                        focus_align.append(cache_transition.focus_manager.get(c))
                        symbol_to_word.append(NONE_WIDX)
                        cache_transition.apply(c, oracle_action)
                        num_promote_actions += 1
                        promote_sym_actions[oracle_action] += 1

                    elif "PROMOTE_ARC:" in oracle_action:
                        _, arclabel = oracle_action.split(":")
                        # Choose arc.
                        feat_seq.append(c.extractFeatures(FeatureType.PROMOTE_ARC,
                            c.next_word_idx, c.next_symbol_data.all, c.cache_size - 1,
                            uniform_arc=uniform))
                        oracle_seq.append(oracle_action)
                        focus_align.append(cache_transition.focus_manager.get(c))
                        cache_transition.apply(c, oracle_action)
                        promote_arc_actions[oracle_action] += 1

                    elif "ARC:" in oracle_action:
                        parts = oracle_action.split(":")
                        arc_decisions = parts[1].split("#")

                        # Maximum possible number of connections
                        num_connect = c.cache_size - 1

                        cache_word_idx, cache_symbol_idx = c.rightmostCache()

                        for cache_idx in range(num_connect - 1, -1, -1):
                            arc_label = arc_decisions[cache_idx]
                            curr_arc_action = "ARC%d:%s" % (cache_idx, arc_label)

                            if arc_label == "O":
                                arc_binary_actions["O"] += 1
                                feat_seq.append(c.extractFeatures(FeatureType.ARCBINARY,
                                                                  cache_word_idx, cache_symbol_idx, cache_idx, uniform_arc=uniform))
                                oracle_seq.append("NOARC")
                                focus_align.append(cache_transition.focus_manager.get(c))
                            else:
                                arc_binary_actions["Y"] += 1
                                feat_seq.append(c.extractFeatures(FeatureType.ARCBINARY,
                                                                  cache_word_idx, cache_symbol_idx, cache_idx, uniform_arc=uniform))
                                oracle_seq.append("ARC")
                                focus_align.append(cache_transition.focus_manager.get(c))

                                feat_seq.append(c.extractFeatures(FeatureType.ARCCONNECT,
                                                                  cache_word_idx, cache_symbol_idx, cache_idx, uniform_arc=uniform))
                                arc_label_actions[arc_label] += 1
                                oracle_seq.append(arc_label)
                                focus_align.append(cache_transition.focus_manager.get(c))

                            cache_transition.apply(c, curr_arc_action)
                        #print("Tree after applying %s: \n %s" %(oracle_action, c.hypothesis))
                    elif "symGen:" in oracle_action:
                        # First add GENSYM action.
                        feat_seq.append(c.extractFeatures(FeatureType.SHIFTPOP,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        oracle_seq.append("GENSYM")
                        focus_align.append(cache_transition.focus_manager.get(c))

                        # Then add SYMSELECT action.
                        feat_seq.append(c.extractFeatures(FeatureType.SYMSELECT,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        oracle_seq.append(oracle_action)
                        focus_align.append(cache_transition.focus_manager.get(c))

                        # Counters
                        num_gensym_actions += 1
                        gensym_actions[oracle_action] += 1

                        # Update symbol->word mapping
                        assert len(symbol_to_word) == c.next_symbol_data.all, "{}\n{}\n{}\n{}".format(len(symbol_to_word), c.next_symbol_data.all, symbol_to_word, c.next_symbol_data.all)
                        symbol_to_word.append(NONE_WIDX)

                        # Apply action to transition system
                        cache_transition.apply(c, oracle_action)

                    elif "seqGen" in oracle_action:
                        # First add SEQGEN action.
                        feat_seq.append(c.extractFeatures(FeatureType.SHIFTPOP,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        oracle_seq.append("SEQGEN")
                        focus_align.append(cache_transition.focus_manager.get(c))

                        # Then add SYMSELECT action.
                        feat_seq.append(c.extractFeatures(FeatureType.SYMSELECT,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        oracle_seq.append(oracle_action)
                        focus_align.append(cache_transition.focus_manager.get(c))

                        # Counters
                        num_seqgen_actions += 1
                        #symbol = oracle_action.split(":")[1]
                        #seqgen_actions[symbol] += 1
                        seqgen_actions[oracle_action] += 1

                        # Update symbol->word mapping
                        assert len(symbol_to_word) == c.next_symbol_data.all, "%d : %d" % (c.next_symbol_data.all, len(symbol_to_word))
                        symbol_to_word.append(NONE_WIDX)

                        # Apply action to transition system
                        cache_transition.apply(c, oracle_action)

                    elif "lemmaGen" in oracle_action or "nameGen" in oracle_action or "tokenGen" in oracle_action:
                        # First add WORDGEN action.
                        feat_seq.append(c.extractFeatures(FeatureType.SHIFTPOP,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        oracle_seq.append("WORDGEN")
                        focus_align.append(cache_transition.focus_manager.get(c))

                        # Choose between name or lemma.
                        feat_seq.append(c.extractFeatures(FeatureType.WORDGEN,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        gen_cat = "LEMMA"
                        if "nameGen" in oracle_action:
                            gen_cat = "NAME"
                        elif "tokenGen" in oracle_action:
                            gen_cat = "TOKEN"
                        oracle_seq.append(gen_cat)
                        focus_align.append(cache_transition.focus_manager.get(c))

                        # Choose suffix.
                        if "lemmaGen" in oracle_action:
                            feat_seq.append(c.extractFeatures(FeatureType.WORDGEN_LEMMA,
                                c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        elif "nameGen" in oracle_action:
                            feat_seq.append(c.extractFeatures(FeatureType.WORDGEN_NAME,
                                c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        elif "tokenGen" in oracle_action:
                            feat_seq.append(c.extractFeatures(FeatureType.WORDGEN_TOKEN,
                                c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        else:
                            raise ValueError()
                        suffix = oracle_action.split(":")[2]
                        oracle_seq.append("SUFFIX:" + suffix)
                        focus_align.append(cache_transition.focus_manager.get(c))

                        # Counters
                        if "lemmaGen" in oracle_action:
                            num_lemmagen_actions += 1
                            lemmagen_actions[oracle_action] += 1
                        elif "nameGen" in oracle_action:
                            num_namegen_actions += 1
                            namegen_actions[oracle_action] += 1
                        else:
                            num_tokengen_actions += 1
                            tokengen_actions[oracle_action] += 1

                        # Update symbol->word mapping and apply action to transition system.
                        symbol_to_word.append(c.peekWordIdx())
                        cache_transition.apply(c, oracle_action)

                    elif "mergeBuf" is oracle_action:
                        feat_seq.append(c.extractFeatures(FeatureType.SHIFTPOP,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        oracle_seq.append("MERGEBUF")
                        focus_align.append(cache_transition.focus_manager.get(c))

                        # Update counters.
                        num_mergebuf_actions += 1

                        cache_transition.apply(c, oracle_action)

                    elif oracle_action == "symID:-NULL-":
                        # Currently assume vertex generated separately.
                        assert c.phase == FeatureType.SHIFTPOP
                        feat_seq.append(c.extractFeatures(FeatureType.SHIFTPOP,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        cache_transition.apply(c, oracle_action)
                        oracle_seq.append(oracle_action)
                        focus_align.append(cache_transition.focus_manager.get(c))
                        if feat_dim == -1:
                            feat_dim = len(feat_seq[-1])
                        # Don't update hyp-gold index (happens after this if-else tre)
                        continue
                    elif oracle_action == "symID:-SKIP-":
                        assert c.phase == FeatureType.SHIFTPOP
                        feat_seq.append(c.extractFeatures(FeatureType.SHIFTPOP,
                            c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                        cache_transition.apply(c, oracle_action)
                        oracle_seq.append(oracle_action)
                        focus_align.append(cache_transition.focus_manager.get(c))
                        if feat_dim == -1:
                            feat_dim = len(feat_seq[-1])
                        # Don't update hyp-gold index (happens after this if-else tre)
                        continue
                    else:
                        assert c.next_symbol_data.all >= 0

                        if not is_symbol_action(oracle_action):
                            focus_align.append(cache_transition.focus_manager.get(c))
                            oracle_seq.append(oracle_action)
                            if oracle_action == "POP":
                                # block out this part
                                feat_seq.append(c.extractFeatures(FeatureType.SHIFTPOP,
                                                                  c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                                num_pop_actions += 1
                            else:
                                feat_seq.append(c.extractFeatures(FeatureType.PUSHIDX,
                                                                  c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                                push_actions[oracle_action] += 1
                        elif "NULL" not in oracle_action:
                            focus_align.append(cache_transition.focus_manager.get(c))
                            if self.unreserve_gen_method == UnreserveGenMethod.NONE:
                                oracle_seq.append("SHIFT")
                            elif self.unreserve_gen_method == UnreserveGenMethod.WORD:
                                oracle_seq.append(oracle_action)
                            else:
                                raise Exception("Unkonwn unreserve_gen_method: {}".format(self.unreserve_gen_method))
                            num_shift_actions += 1
                            assert len(symbol_to_word) == c.next_symbol_data.all, "%d : %d" % (c.next_symbol_data.all, len(symbol_to_word))
                            assert c.next_word_idx >= 0, "c.next_word_idx: {}".format(c.next_word_idx)
                            symbol_to_word.append(c.next_word_idx)

                            feat_seq.append(c.extractFeatures(FeatureType.SHIFTPOP,
                                                              c.next_word_idx, c.next_symbol_data.all, uniform_arc=uniform))
                            if feat_dim == -1:
                                feat_dim = len(feat_seq[-1])
                        else:
                            assert False
                        cache_transition.apply(c, oracle_action)
                        #print("Tree after applying %s: \n %s" %(oracle_action, c.hypothesis))

                    # Update hypo-gold correspondence.
                    if hypo_idx is not None:
                        c.hyp2gold[hypo_idx] = gold_idx
                        c.gold2hyp[gold_idx] = hypo_idx

                if succeed:
                    assert len(feat_seq) == len(oracle_seq), "len(feat_seq): {}\nlen(oracle_seq): {}\nfeat_seq: {}\noracle_seq: {}".format(len(feat_seq), len(oracle_seq), feat_seq, oracle_seq)
                    assert len(oracle_seq) > 0, oracle_seq
                    if not decode and (not oracle_seq):
                        continue
                    if focus_align:
                        assert len(feat_seq) == len(focus_align)
                        assert (focus_align[-1][1] >= cache_transition.buffer_offset and 
                                focus_align[-1][1] <= len(symbol_seq) + cache_transition.buffer_offset), symbol_seq
                        assert focus_align[-1][0] >= 0 and focus_align[-1][0] <= len(inst.tok) + 3, \
                                "focus_align[-1][0]: {}\ninst.tok: {}".format(focus_align[-1][0], inst.tok)
                        assert len(symbol_to_word) == len(symbol_seq), \
                                "len(symbol_to_word): {}\nlen(symbol_seq): {}\nsymbol_to_word: {}\nsymbol_seq: {}".format(len(symbol_to_word), len(symbol_seq), symbol_to_word, symbol_seq)

                    # Add a final feat_seq for predicting the end action.
                    cache_word_idx, cache_symbol_idx = c.rightmostCache()
                    feats = c.extractFeatures(FeatureType.ARCBINARY,
                            cache_word_idx, cache_symbol_idx,
                            c.cache_size - 1, uniform_arc=uniform)
                    feat_seq.append(feats)
                    focus_align.append(cache_transition.focus_manager.get(c))

                    if decode:
                        # TODO: hmm, I think we should keep the feat_seq. These seem sort of important for parsing.
                        #       wait... I think the decoder generates these anyway, so we probably don't need it.
                        feat_seq = [["" for _ in feats] for feats in feat_seq]
                        focus_align = [(NULL_WIDX, NULL_SIDX) for _ in focus_align]

                    oracle_example = OracleExample(inst.tok, inst.lem, inst.pos, inst.ner,
                            c.hypothesis.getSymbolSeq(), map_info_seq, feat_seq,
                            oracle_seq, focus_align, symbol_to_word)

                    # print "feature dimension: %d" % feat_dim
                    for feats in feat_seq:
                        assert len(feats) == feat_dim, "Feature dimensions not consistent. len(feats): {}\nfeat_dim: {}\nfeats: {}".format(len(feats), feat_dim, str(feats))
                    total_oracle_length += len(oracle_seq)
                    if len(oracle_seq) > max_oracle_length:
                        max_oracle_length = len(oracle_seq)
                    oracle_set.addExample(oracle_example)
                    #print("Sentence #%d: %s" % (sent_idx, " ".join(inst.tok)), file=oracle_wf)
                    #print("ULFAMR graph:\n%s" %  str(ulfamr_graph).strip(), file=oracle_wf)
                    #print("Constructed ULFAMR graph:\n%s" % str(c.hypothesis).strip(), file=oracle_wf)

                    oracle_align = " ".join(["%s,%d,%d" % (w_a, c_a, o_s) for (w_a, (c_a, o_s))
                                             in zip(oracle_seq, focus_align)])
                    #print("Oracle sequence: %s" % oracle_align, file=oracle_wf)
                    #print("Oracle sequence: %s" % oracle_seq)
                    for feats in feat_seq:
                        assert len(feats) == feat_dim, "len(feats): {}\nfeat_dim: {}\nfeats: {}\ninst.tok: {}".format(len(feats), feat_dim, feats, inst.tok)

                    #print("visualizing hypo graph now...")
                    #print("hypo graph:")
                    #print(c.hypothesis)
                    #print("visualizing gold graph now...")
                    #print("gold graph:")
                    #print(c.gold)

                    if c.gold.compare(c.hypothesis, c.gold2hyp):
                        success_num += 1
                    else:
                        print("Unsuccessful sentence index %d" %sent_idx)
                        print("Unsuccessful Oracle", file=oracle_wf)
                        print("Gold ULFAMR graph:\n%s" % str(c.gold).strip(), file=oracle_wf)
                        raise Exception('Unsuccessful oracle')

                else:
                    print("Failed sentence %d" % sent_idx)
                    print(" ".join(inst.tok))
                    print(str(ulfamr_graph))
                    print("Oracle sequence so far: %s\n" % " ".join(oracle_seq))
                    raise Exception('Unsuccessful oracle')

            oracle_wf.close()

        if not decode:
            arc_binary_path = os.path.join(output_dir, "arc_binary_actions.txt")
            saveCounter(arc_binary_actions, arc_binary_path)
            arc_label_path = os.path.join(output_dir, "arc_label_actions.txt")
            saveCounter(arc_label_actions, arc_label_path)
            pushidx_path = os.path.join(output_dir, "pushidx_actions.txt")
            saveCounter(push_actions, pushidx_path)
            gensym_path = os.path.join(output_dir, "gensym_actions.txt")
            saveCounter(gensym_actions, gensym_path)
            seqgen_path = os.path.join(output_dir, "seqgen_actions.txt")
            saveCounter(seqgen_actions, seqgen_path)
            promote_binary_path = os.path.join(output_dir, "promote_binary_actions.txt")
            saveCounter(promote_binary_actions, promote_binary_path)
            promote_sym_path = os.path.join(output_dir, "promote_sym_actions.txt")
            saveCounter(promote_sym_actions, promote_sym_path)
            promote_arc_path = os.path.join(output_dir, "promote_arc_actions.txt")
            saveCounter(promote_arc_actions, promote_arc_path)
        print("A total of %d shift actions" % num_shift_actions)
        print("A total of %d pop actions" % num_pop_actions)
        print("A total of %d gensym actions" % num_gensym_actions)
        print("A total of %d seqgen actions" % num_seqgen_actions)
        print("A total of %d lemmagen actions" % num_lemmagen_actions)
        print("A total of %d namegen actions" % num_namegen_actions)
        print("A total of %d tokengen actions" % num_tokengen_actions)
        print("A total of %d mergebuf actions" % num_mergebuf_actions)
        print("A total of %d promote actions" % num_promote_actions)
        print("Maximum oracle sequence length is", max_oracle_length)
        print("Average oracle sequence length is", total_oracle_length/data_num)
        print("Oracle success ratio is", success_num/data_num)
        print("feature dimensions:", feat_dim)
        print("Number of examples: ", oracle_set.num_examples)
        print("generating oracle json")
        oracle_set.toJSON(json_output)
        #print("Oracle sequence: %s" %oracle_seq)

    def isPredicate(self, s):
        length = len(s)
        if length < 3 or not (s[length-3] == '-'):
            return False
        last_char = s[-1]
        return last_char >= '0' and last_char <= '9'

    def symbolCategory(self, s, symbolArcChoices):
        """
        To be implemented!
        :param s:
        :param symbolArcChoices:
        :return:
        """
        if s in symbolArcChoices:
            return s
        if s == "NE" or "NE_" in s:
            return "NE"
        return "OTHER"

    def getWordID(self, s):
        if s in self.wordIDs:
            return self.wordIDs[s]
        return UNKNOWN

    def getLemmaID(self, s):
        if s in self.lemmaIDs:
            return self.lemmaIDs[s]
        return UNKNOWN

    def getSymbolID(self, s):
        if s in self.symbolIDs:
            return self.symbolIDs[s]
        return UNKNOWN

    def getPOSID(self, s):
        if s in self.posIDs:
            return self.posIDs[s]
        return UNKNOWN

    def getNERID(self, s):
        if s in self.nerIDs:
            return self.nerIDs[s]
        return UNKNOWN

    def getDepID(self, s):
        if s in self.depIDs:
            return self.depIDs[s]
        return UNKNOWN

    def getArcID(self, s):
        if s in self.arcIDs:
            return self.arcIDs[s]
        return UNKNOWN

    def actionType(self, s):
        if s == "POP" or any([x in s for x in ["symID", "symGen", "symMod"]]):
            return 0
        if "ARC" in s:
            return 1
        else:
            return 2

    def generateTrainingExamples(self):
        return

def is_symbol_action(action_str):
    return any([
        act_class in action_str
        for act_class
        in ["symID", "symMod", "symGen", "seqGen"]
    ])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, help="The data directory for the input files.")
    argparser.add_argument("--output_dir", type=str, help="The directory for the output files.")
    argparser.add_argument("--cache_size", type=int, default=6, help="Fixed cache size for the transition system.")
    argparser.add_argument("--decode", action="store_true", help="if to extract decoding examples.")
    argparser.add_argument("--uniform", action="store_true", default=False, help="if to use uniform arc features.")
    argparser.add_argument("--ulf", action="store_true", help="if this is for ULFs.")
    argparser.add_argument("--predicted_symbols", action="store_true", help="Whether to use predicted symbols.")
    argparser.add_argument("--pred_symbol_path", type=str, help="Path to predicted symbols.")
    argparser.add_argument("--inseq_symbol_file", type=str, default=None, help="Path to file with INSEQ generation-reserved symbols.")
    argparser.add_argument("--promote_symbol_file", type=str, default=None, help="Path to file with PROMOTE generation-reserved symbols.")
    argparser.add_argument("--reserve_gen_method", type=str, default="none", choices=["none", "inseq", "promote"],
            help="Method to generate reserved symbols")
    argparser.add_argument("--unreserve_gen_method", type=str, default="none", choices=["none", "word"],
            help="Method to generate unreserved symbols. 'none' starts with symbol indices, 'word' generates them from the input words.")
    argparser.add_argument("--focus_method", type=str, default="none", choices=["none", "cache", "triple"],
            help="Method to generate word and symbol focuses. 'none' always uses next generation index, 'cache' generates them from the cache as appropriate, 'triple' generates a static triple of the two cache entries and the next generation index.")
    argparser.add_argument("--buffer_offset", type=int, default=0, help="Offset of the buffer (symbol) indices for the neural network input (for start of sequence or unk tokens).")
    args = argparser.parse_args()

    reserve_gen_method = ReserveGenMethod.NONE
    if args.reserve_gen_method == "inseq":
        reserve_gen_method = ReserveGenMethod.INSEQ
    elif args.reserve_gen_method == "promote":
        reserve_gen_method = ReserveGenMethod.PROMOTE

    # inseq_symbol_file iff reserve_gen_method is not NONE
    assert (args.inseq_symbol_file is None) == (reserve_gen_method == ReserveGenMethod.NONE)
    assert (args.promote_symbol_file is not None) == (reserve_gen_method == ReserveGenMethod.PROMOTE)

    unreserve_gen_method = UnreserveGenMethod.NONE
    if args.unreserve_gen_method == "word":
        unreserve_gen_method = UnreserveGenMethod.WORD
    focus_method = FocusMethod.NONE
    if args.focus_method == "cache":
        focus_method = FocusMethod.CACHE
    elif args.focus_method == "triple":
        focus_method = FocusMethod.TRIPLE

    parser = CacheTransitionParser(
        args.cache_size,
        args.inseq_symbol_file,
        args.promote_symbol_file,
        reserve_gen_method,
        unreserve_gen_method,
        focus_method,
        args.buffer_offset,
    )
    parser.OracleExtraction(
        args.data_dir,
        args.output_dir,
        args.decode,
        args.uniform,
        args.ulf,
        args.predicted_symbols,
        args.pred_symbol_path,
    )

import json
import re, os, sys
import oracle.ioutil
import numpy as np
import random
from collections import defaultdict, namedtuple
from oracle.utils import ReserveGenMethod, UnreserveGenMethod, loadTokens


DataInstance = namedtuple('DataInstance',
    [
        'sentence',  # token sequence
        'symbols_idx',   # symbol sequence (indices)
        'sid2wid',   # symbol to word mapping
        'features_idx',   # features (list for each symbol)
        'actions_idx',  # oracle action indices
        'action2sid',  # action to symbol id
        'action2wid',  # action to word mapping
    ]
)


class AnonSentence(object):
    def __init__(self, tok, lemma, pos, ner, symbols=None, map_info=None,
                 isLower=False, dep=None, idx=-1):
        self.tokText = tok
        self.lemma = lemma
        self.pos = pos
        self.ner = ner
        # it's the answer sequence
        if isLower:
            self.tokText = self.tokText.lower()
            self.lemma = self.lemma.lower()
        self.tok = re.split("\\s+", self.tokText)
        self.lemma = re.split("\\s+", self.lemma)
        self.pos = re.split("\\s+", self.pos)
        self.ner = re.split("\\s+", self.ner)
        self.symbols = symbols
        self.map_info = map_info
        self.tree = dep
        self.length = len(self.tok)
        # Sentence index.
        self.idx = idx
        self.index_convered = False

    def get_length(self):
        return self.length

    def get_max_word_len(self):
        max_word_len = 0
        for word in self.tok:
            max_word_len = max(max_word_len, len(word))
        return max_word_len

    def get_char_len(self):
        return [len(word) for word in self.tok]


def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines


def read_all_GenerationDatasets(inpath, isLower=True, dep_path=None, token_path=None, ulfdep=False, pred_symbol_path=False, pred_symbol_align=False, pred_sid2wid=False, debug=False, presplit_fields=False, outer_feat_delim=" "):
    with open(inpath) as dataset_file:
        dataset = json.load(dataset_file, encoding='utf-8')
    all_instances = []
    trees, tok_seqs = None, []
    if pred_symbol_path:
        pred_symbols = loadTokens(pred_symbol_path, delim="\t")
        pred_aligns = loadTokens(pred_symbol_align, delim="_#_")
        pred_sid2wids = json.load(open(pred_sid2wid, 'r'))
    sent_idx = 0
    for instance in dataset:
        text = instance['text']
        if dep_path:
            tok_seqs.append(text.split())
        lemma = instance['annotation']['lemmas']
        pos = instance['annotation']['POSs']
        ner = instance['annotation']['NERs']
        if pred_symbol_path:
            symbols = pred_symbols[sent_idx]
            map_infos = pred_aligns[sent_idx]
        else:
            symbols = (
                instance['symbol_tokens']
                if presplit_fields
                else instance['symbols'].strip().split()
            )
            map_infos = instance['annotation']['mapinfo'].split("_#_")
        assert len(symbols) == len(map_infos), "len(symbols): {}\nlen(map_infos): {}\nsymbols: {}\nmap_infos: {}".format(
                len(symbols), len(map_infos), symbols, map_infos)

        input_sent = AnonSentence(text, lemma, pos, ner, symbols, map_infos, idx=sent_idx)

        feats = [x.split('_#_') for x in instance['feats'].strip().split(outer_feat_delim)]
        #print("feature dimension: %d" % len(feats[0]))
        actions = instance['actions'] if presplit_fields else instance['actionseq'].strip().split()
        action2sid = [int(x) for x in instance['alignment']['symbol-align'].strip().split()]
        action2wid = [int(x) for x in instance['alignment']['word-align'].strip().split()]
        if pred_symbol_path:
            sid2wid = pred_sid2wids[sent_idx]
        else:
            sid2wid = [int(x) for x in instance['alignment']['symbol-to-word'].strip().split()]

        if debug and text == "enjoy your meal !":
            print("text: {}".format(text))
            print("lemma: {}".format(lemma))
            print("pos: {}".format(pos))
            print("ner: {}".format(ner))
            print("symbols: {}".format(symbols))
            for i in range(50):
                print("")
                print("action: {}".format(actions[i]))
                print("feats: {}".format(feats[i]))
            import sys
            sys.exit()

        sent_idx += 1

        if dep_path is None and len(feats) == 0:
            continue

        assert len(feats) == len(actions) + 1
        assert len(feats) == len(action2sid)
        assert len(feats) == len(action2wid)

        # TODO: Check whether add sid2wid changes some parts of training!
        all_instances.append(DataInstance(input_sent, symbols, sid2wid, feats, actions, action2sid, action2wid))
    if dep_path:
        all_orig_tokens = loadTokens(token_path)
        assert len(dataset) == len(all_orig_tokens), \
                "len(dataset): {}\nlen(all_orig_tokens): {}".format(len(dataset), len(all_orig_tokens))
        for (sent_idx, orig_seq) in enumerate(all_orig_tokens):
            assert len(orig_seq) == len(tok_seqs[sent_idx]), "len(orig_seq): {}\nlen(tok_seqs[sent_idx]): {}\norig_seq: {}\ntok_seqs[sent_idx]: {}]\nsent_idx: {}".format(
                    len(orig_seq), len(tok_seqs[sent_idx]), orig_seq, tok_seqs[sent_idx], sent_idx)
        # trees = oracle.ioutil.loadDependency(dep_path, tok_seqs, True)
        trees = oracle.ioutil.loadDependency(dep_path, all_orig_tokens, ulfdep=ulfdep)

        assert len(trees) == len(all_instances), ("inconsistent number of dependencies and instances: "
                                                  "%d vs %d" % (len(trees), len(all_instances)))
        for (idx, tree) in enumerate(trees):
            all_instances[idx][0].tree = tree

    return all_instances

def load_actions(indir, reserve_gen_method=ReserveGenMethod.NONE, unreserve_gen_method=UnreserveGenMethod.NONE):
    shiftpop_list = ["SHIFT", "POP"]
    if reserve_gen_method == ReserveGenMethod.INSEQ:
        shiftpop_list.extend(["GENSYM", "SEQGEN"])
    if unreserve_gen_method == UnreserveGenMethod.WORD:
        shiftpop_list.extend(["MERGEBUF", "WORDGEN", "symID:-NULL-", "symID:-SKIP-"])
    shiftpop_actions = set(shiftpop_list)
    pushidx_actions = oracle.ioutil.loadCounter(os.path.join(indir, "pushidx_actions.txt"))
    arcbinary_actions = set(["NOARC", "ARC"])
    arclabel_actions = oracle.ioutil.loadCounter(os.path.join(indir, "arc_label_actions.txt"))
    gensym_actions = oracle.ioutil.loadCounter(os.path.join(indir, "gensym_actions.txt"))
    seqgen_actions = oracle.ioutil.loadCounter(os.path.join(indir, "seqgen_actions.txt"))
    promote_sym_actions = oracle.ioutil.loadCounter(os.path.join(indir, "promote_sym_actions.txt"))
    promote_arc_actions = oracle.ioutil.loadCounter(os.path.join(indir, "promote_arc_actions.txt"))
    return shiftpop_actions, pushidx_actions, arcbinary_actions, arclabel_actions, gensym_actions, seqgen_actions, promote_sym_actions, promote_arc_actions

def load_arc_choices(indir):
    # TODO: figure out how to generate these files dammit...
    #outgo_arc_choices = oracle.ioutil.loadArcMaps(os.path.join(indir, "symbol_rels.txt"))
    #income_arc_choices = oracle.ioutil.loadArcMaps(os.path.join(indir, "symbol_incomes.txt"))
    outgo_arc_choices = defaultdict(set)
    income_arc_choices = defaultdict(set)
    default_arc_choices = oracle.ioutil.defaultArcChoices()
    print ("ARC choices for %d incomes, %d outgos, %d defaults." % (len(income_arc_choices),
                                                                    len(outgo_arc_choices), len(default_arc_choices)))
    return income_arc_choices, outgo_arc_choices, default_arc_choices

def read_Testset(indir, isLower=True, decode=False, ulfdep=False, use_pred_symbols=False):
    if decode:
        test_file = os.path.join(indir, "decode.json")
    else:
        test_file = os.path.join(indir, "oracle_examples.json")
    dep_file = os.path.join(indir, "dep")
    token_file = os.path.join(indir, "token")
    if use_pred_symbols:
        pred_symbol_file = os.path.join(indir, "symbol.pred")
        pred_symbol_align = os.path.join(indir, "alignments.pred")
        pred_atom2word = os.path.join(indir, "atom2word.pred")
        return read_all_GenerationDatasets(test_file, isLower, dep_file, token_file, ulfdep=ulfdep, pred_symbol_path=pred_symbol_file, pred_symbol_align=pred_symbol_align, pred_sid2wid=pred_atom2word)
    else:
        return read_all_GenerationDatasets(test_file, isLower, dep_file, token_file, ulfdep=ulfdep)

def read_generation_datasets_from_fof(fofpath, isLower=True, ulfdep=False):
    all_paths = read_text_file(fofpath)
    all_instances = []
    for cur_path in all_paths:
        print(cur_path)
        cur_instances = read_all_GenerationDatasets(cur_path, isLower=isLower, ulfdep=ulfdep)
        all_instances.extend(cur_instances)
    return all_instances


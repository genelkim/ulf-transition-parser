#!/usr/bin/python3
import re, os
import sys
from collections import defaultdict
from enum import Enum

class ReserveGenMethod(Enum):
    """Methods for generating the reserved symbols.
    NONE: no generation -- assume all symbols are provided
    INSEQ: generate in preorder sequence, just generate and treat as other symbols
    PROMOTE: promote an existing completed node to the reserved symbol
    """
    NONE = 1
    INSEQ = 2
    PROMOTE = 3

class UnreserveGenMethod(Enum):
    NONE = 1
    WORD = 2

class FocusMethod(Enum):
    """
    NONE: just use the next word and symbol indices
    CACHE: use rightmost cache as appropriate (ARCs)
    TRIPLE: alway both cache indices and next word
    """
    NONE = 1
    CACHE = 2
    TRIPLE = 3

class TypeCheckingMethod(Enum):
    """Methods for checking that the ULF type constraints are being followed
    appropriately.
    """
    NONE = 1
    COMPOSITION = 2
    SANITY_CHECKER = 3

RESERVE_GEN_NAMES = {
    "none": ReserveGenMethod.NONE,
    "inseq": ReserveGenMethod.INSEQ,
    "promote": ReserveGenMethod.PROMOTE,
}
UNRESERVE_GEN_NAMES = {
    "none": UnreserveGenMethod.NONE,
    "word": UnreserveGenMethod.WORD,
}
FOCUS_NAMES = {
    "none": FocusMethod.NONE,
    "cache": FocusMethod.CACHE,
    "triple": FocusMethod.TRIPLE,
}
TYPE_CHECKING_NAMES = {
    "none": TypeCheckingMethod.NONE,
    "composition": TypeCheckingMethod.COMPOSITION,
    "sanity_checker": TypeCheckingMethod.SANITY_CHECKER,
}

class TokenType(Enum):
    WORD = 1
    LEM = 2
    POS = 3
    SYMBOL = 4
    DEP = 5
    ARC = 6
    NER = 7

class FeatureType(Enum):
    SHIFTPOP = 1
    ARCBINARY = 2
    ARCCONNECT = 3
    PUSHIDX = 4
    SYMSELECT = 5
    WORDGEN = 6
    WORDGEN_LEMMA = 7
    WORDGEN_NAME = 8
    WORDGEN_TOKEN = 9
    PROMOTE = 10
    PROMOTE_ARC = 11
    PROMOTE_SYM = 12

class ArcDir(Enum):
    LEFT = 0
    RIGHT = 1

tokentype_feat_num = 5
symbol_relation_feat_num = 4

# right cache token feats, buffer token feats, dependency features, symbol features
def shiftpop_feat_num():
    return tokentype_feat_num * 2 + 4 + 4
# candidate token feats, cache token feats, candidate arc feats, cache arc feats,  distance feats
def cache_feat_num():
    return 2 * tokentype_feat_num + 2 * symbol_relation_feat_num + 4
# candidate token feats, each cache index token feats
def pushidx_feat_num(cache_size):
    return tokentype_feat_num * (1 + cache_size)
# rightmost cache token feats, widx buffer token feats
def symselect_feat_num():
    return tokentype_feat_num * 2
def promote_feat_num():
    #return shiftpop_feat_num()
    return cache_feat_num()

# Total number of features for each action.
# See oracle.cacheConfiguration.extractFeatures to see the actual computation of the feature list.
def action_feat_num(cache_size):
    return 1 + shiftpop_feat_num() + ((2 * cache_feat_num()) + 8) + pushidx_feat_num(cache_size) + symselect_feat_num() + ((2 * promote_feat_num()) + 8)


NULL = "-NULL-"
UNKNOWN = "-UNK-"

# These are not the actual indices, but offsets that are used by the focus
# manager in conjunction with the configuration state to get the actual
# index in the network architecture.
END_WIDX = -1
NONE_WIDX = -2
NULL_WIDX = -3
NULL_SIDX = -1


symbols = set("'\".-!#&*|\\/")
re_symbols = re.compile("['\".\'`\-!#&*|\\/@=\[\]]")
special_token_map = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}

def allSymbols(s):
    return re.sub(re_symbols, "", s) == ""

def equals(s1, s2):
    s2_sub = s2.replace("-RRB-", ")").replace("-LRB-", "(")
    s2_sub = re.sub(re_symbols, "", s2_sub)
    s1 = s1.replace("\xc2\xa3", "#")
    s1_sub = re.sub(re_symbols, "", s1)

    return s1_sub.lower() == s2_sub.lower()

def in_bracket(start, end, tok_seq):
    length = len(tok_seq)
    has_left = False
    for idx in range(start-3, start):
        if idx >= 0 and tok_seq[idx] == "(":
            has_left = True
    if not has_left:
        return False
    for idx in range(end, end+3):
        if idx < length and tok_seq[idx] == ")":
            return True
    return False

def loadTokens(path, delim=" "):
    """Reads in tokens assuming each example is in a single line with given delimiter between tokens.
    """
    tok_seqs = []
    with open(path, "r", encoding='utf-8') as tok_f:
        for line in tok_f:
            line = line.strip()
            if not line:
                tok_seqs.append([])
            else:
                toks = line.split(delim)
                tok_seqs.append(toks)
    return tok_seqs

def loadAlignedTokens(path, example_delim="\n\n", token_delim="\n", align_delim="\t", align_idx=1):
    """Reads in tokens where each token is aligned with the token annotation.
    """
    tok_seqs = []
    filestr = open(path, mode='r').read()
    for ex in filestr.split(example_delim):
        ex = ex.strip()
        aligned_toks = ex.split(token_delim)
        toks = [a_tok.strip().split(align_delim)[align_idx] for a_tok in aligned_toks]
        tok_seqs.append(toks)
    return tok_seqs

def loadNERTokens(path):
    return loadAlignedTokens(path, "\n\n", "\n", "\t", 1)

def loadDepTokens(dep_file):
    ret = []
    with open(dep_file, "r") as dep_f:
        sent_idx = 0
        tok_seq = []
        for line in dep_f:
            splits = line.strip().split("\t")
            if len(splits) < 10:
                if tok_seq:
                    ret.append(tok_seq)
                tok_seq = []
                sent_idx += 1
            else:
                word = splits[1]
                for sp in special_token_map:
                    if sp in word:
                        word = word.replace(sp, special_token_map[sp])
                # if word in special_token_map:
                #     word = special_token_map[word]
                tok_seq.append(word)
        dep_f.close()
    return ret

def saveMLE(counts, dists, path, used_set=None, verify=False):
    def check(toks, node_repr):
        for s in toks:
            if s not in node_repr:
                return False
        return True
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    with open(path, "w") as wf:
        for (item ,count) in sorted_items:
            if count == 1:
                if verify and not check(item.split(), dists[item].items()[0][0]):
                    continue

            sorted_repr = sorted(dists[item].items(), key=lambda x: -x[1])
            dist_repr = ";".join(["%s:%d" % (s, c) for (s, c) in sorted_repr])
            if used_set and item not in used_set:
                print("Filtered phrase:", item)
                continue
            print("%s\t%d\t%s" % (item, count, dist_repr), file=wf)
        wf.close()

def loadCountTable(path, max_len=3):
    counts = {}
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                fields = line.strip().split("\t")
                if len(fields[0].split()) > max_len:
                    print("Pruned phrase:", fields[0])
                    continue
                counts[fields[0]] = int(fields[1])
    return counts

def saveCounter(counts, path):
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    with open(path, "w") as wf:
        for (item ,count) in sorted_items:
            print("%s\t%d" % (item, count), file=wf)
        wf.close()

def saveSetorList(entities, path):
    with open(path, "w") as wf:
        for item in entities:
            print(item, file=wf)
        wf.close()

def loadMLEFile(path):
    mle_map = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                splits = line.strip().split("####")
                word = splits[0]
                category = splits[1].split("##")[0]
                node_repr = splits[1].split("##")[1]
                mle_map[word] = (category, node_repr)
    return mle_map

def loadPhrases(path):
    mle_map = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                splits = line.strip().split("\t")
                phrase = splits[0]
                node_repr = splits[2].split("#")[0].split(":")[0]
                mle_map[phrase] = node_repr
    return mle_map

#Build the entity map for symbol identification
#Choose either the most probable category or the most probable node repr
#In the current setting we only care about NE and DATE
def loadMap(map_file):
    span_to_cate = {}

    #First load all possible mappings each span has
    with open(map_file, 'r') as map_f:
        for line in map_f:
            if line.strip():
                spans = line.strip().split('##')
                for s in spans:
                    try:
                        fields = s.split('++')
                        toks = fields[1]
                        wiki_label = fields[2]
                        node_repr = fields[3]
                        category = fields[-1]
                    except:
                        print(spans, line)
                        print(fields)
                        sys.exit(1)
                    if toks not in span_to_cate:
                        span_to_cate[toks] = defaultdict(int)
                    span_to_cate[toks][(category, node_repr, wiki_label)] += 1

    mle_map = {}
    for toks in span_to_cate:
        sorted_types = sorted(span_to_cate[toks].items(), key=lambda x:-x[1])
        curr_type = sorted_types[0][0][0]
        if curr_type[:2] == 'NE' or curr_type[:4] == 'DATE':
            mle_map[toks] = sorted_types[0][0]
    return mle_map

def dumpMap(mle_map, result_file):
    with open(result_file, 'w') as wf:
        for toks in mle_map:
            print(('%s####%s##%s' % (toks, mle_map[toks][0], mle_map[toks][1])), file=wf)

def dateMap(dateFile):
    dates_in_lines = []
    for line in open(dateFile):
        date_spans = []
        if line.strip():
            spans = line.strip().split()
            for sp in spans:
                start = int(sp.split('-')[0])
                end = int(sp.split('-')[1])
                date_spans.append((start, end))
        dates_in_lines.append(date_spans)
    return dates_in_lines

def alignDates(tok_file, dateFile):
    tok_seqs = loadTokens(tok_file)
    date_spans = loadTokens(dateFile)
    assert len(tok_seqs) == len(date_spans)
    for (idx, dates) in enumerate(date_spans):
        toks = tok_seqs[idx]
        for sp in dates:
            start = int(sp.split("-")[0])
            end = int(sp.split("-")[1])
            print(" ".join(toks[start:end]))

def check_tokenizer(input_file, tokenized_file):
    with open(input_file, 'r') as input_f:
        with open(tokenized_file, 'r') as tokenized_f:
            for input_line in input_f:
                tokenized_line = tokenized_f.readline()
                if input_line.strip():
                    input_repr = "".join(input_line.strip().split()).replace("\"", "").replace("\'", "")
                    tokenized_repr = "".join(tokenized_line.strip().split()).replace("\'", "")
                    if input_repr != tokenized_repr:
                        print("original tokens:", input_line.strip())
                        print("tokenized:", tokenized_line.strip())

def loadFrequentSet(path):
    symbol_path = os.path.join(path, "symbol_counts.txt")
    frequent_symbol = loadbyfreq(symbol_path, 500)

    symboloutgo_path = os.path.join(path, "symbol_rels.txt")
    frequent_outgo = loadbyfreq(symboloutgo_path, 100)

    symbolincome_path = os.path.join(path, "symbol_incomes.txt")
    frequent_income = loadbyfreq(symbolincome_path, 100)

    return frequent_symbol | (frequent_income & frequent_outgo)

def getCategories(token, frequent_set, verify=False):
    if verify:
        assert token != "NONE"

    if token in frequent_set:
        return token
    if token == "NONE":
        return token
    if token == "MULT" or "MULT_" in token:
        assert "MULT_" in token, token
        return getCategories(token[5:], frequent_set)
    if token == "NEG" or "NEG_" in token:
        assert "NEG_" in token, token
        return getCategories(token[4:], frequent_set)
    if token == "NE" or "NE_" in token:
        return "NE"
    if token == "NUMBER" or token == "DATE":
        return token
    if re.match(".*-[0-9]+", token) is not None:
        return "PRED"
    return "OTHER"

def special_categories(symbol):
    if symbol in set(["NUMBER", "DATE", "NE"]):
        return True
    return symbol[:4] == "MULT" or symbol[:3] == "NEG" or symbol[:3] == "NE_"

def entity_category(symbol):
    if symbol in set(["NUMBER", "DATE", "NE"]):
        return True
    return symbol[:3] == "NE_"

def loadbyfreq(file, threshold, delimiter="\t"):
    curr_f = open(file, 'r')
    symbol_set = set()
    for line in curr_f:
        if line.strip():
            fields = line.strip().split(delimiter)
            symbol = fields[0]
            curr_freq = int(fields[1])
            if curr_freq >= threshold:
                symbol_set.add(symbol)
    return symbol_set

def saveSpanMap(span_lists, path, delimiter="####"):
    with open(path, "w") as wf:
        for span_seq in span_lists:
            new_span_seq = sorted([(int(s.split(delimiter)[0].split("-")[0]),
                                    int(s.split(delimiter)[0].split("-")[1]), s) for s in span_seq])
            print("\t".join([s for (_, _, s) in new_span_seq]), file=wf)
        wf.close()

def loadSpanMap(path, delimiter="\t", type="NER"):
    all_maps = []
    with open(path, 'r') as f:
        for line in f:
            span_map = {}
            if not line.strip():
                all_maps.append(span_map)
                continue
            for map_str in line.strip().split(delimiter):
                span_str = map_str.split("####")[0]
                start = int(span_str.split("-")[0])
                end = int(span_str.split("-")[1])

                symbol_str = map_str.split("####")[1].strip()
                if type == "NER":
                    ne_type = symbol_str.split("##")[0].strip()
                    wiki_str = symbol_str.split("##")[1].strip()
                    span_map[(start, end)] = (ne_type, wiki_str)
                else: #DATE
                    span_map[(start, end)] = symbol_str
            all_maps.append(span_map)
        return all_maps


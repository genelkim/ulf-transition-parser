import sys, os
from .dependency import DependencyTree
from .ULFAMRGraph import *
NULL = "-NULL-"
UNKNOWN = "-UNK-"
from .utils import loadDepTokens, loadTokens, loadAlignedTokens, loadNERTokens
from collections import namedtuple

DatasetInstance = namedtuple('DatasetInstance', ['tok', 'lem', 'pos', 'ner', 'dep', 'ulfamr'])

class Dataset(object):
    def __init__(self, tok_thre=1, top_arc=50):
        self.tok_seqs = None
        self.lem_seqs = None
        self.pos_seqs = None
        self.ner_seqs = None
        self.dep_trees = None
        self.ulfamr_graphs = None

        self.tok_threshold = tok_thre
        self.top_arc_num = top_arc

        self.known_toks = None
        self.known_lems = None
        self.known_poss = None
        self.known_ners = None
        self.known_deps = None
        self.known_symbols = None
        self.known_rels = None
        self.all_labels = None

        self.tokIDs = {}
        self.lemIDs = {}
        self.posIDs = {}
        self.nerIDs = {}
        self.depIDs = {}
        self.symbolIDs = {}
        self.relIDs = {}

        self.tok_offset = 0
        self.lem_offset = 0
        self.pos_offset = 0
        self.ner_offset = 0
        self.dep_offset = 0
        self.symbol_offset = 0
        self.rel_offset = 0

        self.unaligned_set = set()

    def setTok(self, tok_seqs):
        self.tok_seqs = tok_seqs

    def setLemma(self, lem_seqs):
        self.lem_seqs = lem_seqs

    def setPOS(self, pos_seqs):
        self.pos_seqs = pos_seqs

    def setNER(self, ner_seqs):
        self.ner_seqs = ner_seqs

    def setDepTrees(self, dep_trees):
        self.dep_trees = dep_trees

    def setULFAMRGraphs(self, ulfamr_graphs):
        self.ulfamr_graphs = ulfamr_graphs

    def getInstance(self, idx):
        return DatasetInstance(
            self.tok_seqs[idx],
            self.lem_seqs[idx],
            self.pos_seqs[idx],
            self.ner_seqs[idx],
            self.dep_trees[idx],
            self.ulfamr_graphs[idx],
        )

    def dataSize(self):
        return len(self.tok_seqs)

    def genDictionaries(self):
        """
        Given the tokens and labels of the dataset,
        generate the dictionaries for this dataset.
        :return:
        """
        def flattenedSeq(all_seqs):
            labels = []
            for seq in all_seqs:
                for tok in seq:
                    labels.append(tok)
            return labels

        def depLabels(all_trees):
            labels = []
            for tree in all_trees:
                for l in tree.label_list:
                    labels.append(l)
            return labels

        def ulfamrLabels(all_graphs):
            symbol_labels = []
            arc_labels = []
            for graph in all_graphs:
                for symbol in graph.symbols:
                    symbol_labels.append(symbol.getValue())
                    for l in symbol.rels:
                        arc_labels.append(l)
            return symbol_labels, arc_labels

        tok_list = flattenedSeq(self.tok_seqs)
        lem_list = flattenedSeq(self.lem_seqs)
        pos_list = flattenedSeq(self.pos_seqs)
        ner_list = flattenedSeq(self.ner_seqs)
        dep_list = depLabels(self.dep_trees)
        symbol_list, arc_list = ulfamrLabels(self.ulfamr_graphs)

        # Generate the label list.
        self.known_toks = generateDictionary(tok_list, self.tok_threshold)
        self.known_lems = generateDictionary(lem_list)
        self.known_poss = generateDictionary(pos_list)
        self.known_ners = generateDictionary(ner_list)
        self.known_deps = generateDictionary(dep_list)
        self.known_symbols = generateDictionary(symbol_list)
        self.known_rels = generateDictionary(arc_list)

        self.known_toks.insert(0, UNKNOWN)
        self.known_lems.insert(0, UNKNOWN)
        self.known_poss.insert(0, UNKNOWN)
        self.known_ners.insert(0, UNKNOWN)
        self.known_deps.insert(0, UNKNOWN)
        self.known_symbols.insert(0, UNKNOWN)
        self.known_rels.insert(0, UNKNOWN)

        self.known_toks.insert(1, NULL)
        self.known_lems.insert(1, NULL)
        self.known_poss.insert(1, NULL)
        self.known_ners.insert(1, NULL)
        self.known_deps.insert(1, NULL)
        self.known_symbols.insert(1, NULL)
        self.known_rels.insert(1, NULL)

        self.all_labels = self.known_toks + self.known_lems + self.known_poss + self.known_ners + \
                self.known_deps + self.known_symbols + self.known_rels

        self.generateIDs()

        print("#Tokens: %d" % len(self.known_toks), file=sys.stderr)
        print("#Lemmas: %d" % len(self.known_lems), file=sys.stderr)
        print("#POSs: %d" % len(self.known_poss), file=sys.stderr)
        print("#NERs: %d" % len(self.known_ners), file=sys.stderr)
        print("#Dependency labels: %d" % len(self.known_deps), file=sys.stderr)
        print("#Symbols: %d" % len(self.known_symbols), file=sys.stderr)
        print("#Relations: %d" % len(self.known_rels), file=sys.stderr)

    def generateIDs(self):
        for idx, tok in enumerate(self.known_toks):
            self.tokIDs[tok] = idx
        self.lem_offset = len(self.known_toks)
        for idx, lem in enumerate(self.known_lems):
            self.lemIDs[lem] = self.lem_offset + idx
        self.pos_offset = self.lem_offset + len(self.known_lems)
        for idx, pos in enumerate(self.known_poss):
            self.posIDs[pos] = self.pos_offset + idx
        self.ner_offset = self.pos_offset + len(self.known_poss)
        for idx, ner in enumerate(self.known_ners):
            self.nerIDs[ner] = self.ner_offset + idx
        self.dep_offset = self.ner_offset + len(self.known_ners)
        for idx, dep in enumerate(self.known_deps):
            self.depIDs[dep] = self.dep_offset + idx
        self.symbol_offset = self.dep_offset + len(self.known_deps)
        for idx, symbol in enumerate(self.known_symbols):
            self.symbolIDs[symbol] = self.symbol_offset + idx
        self.rel_offset = self.symbol_offset + len(self.known_symbols)
        for idx, rel in enumerate(self.known_rels):
            self.relIDs[rel] = self.rel_offset + idx
            assert self.all_labels[self.rel_offset+idx] == rel

    def genSymbolMap(self):
        """
        To be updated!
        :return:
        """
        return

    def saveSymbolID(self, path):
        """
        To be updated!
        :param path:
        :return:
        """
        return

    def saveMLESymbolID(self, path):
        """
        To be updated!
        :param path:
        :return:
        """
        return

# Return a list rather than a dictionary?
def generateDictionary(label_list, thre=1):
    counts = defaultdict(int)
    for l in label_list:
        counts[l] += 1
    new_label_list = []
    sorted_label_list = sorted(counts.items(), key=lambda x: -x[1])
    for (l, count) in sorted_label_list:
        if count < thre:
            break
        if l == UNKNOWN or l == NULL:
            continue
        new_label_list.append(l)
    return new_label_list

def topkDictionary(label_list, k):
    counts = defaultdict(int)
    for l in label_list:
        counts[l] += 1
    new_label_list = []
    sorted_label_list = sorted(counts.items(), key=lambda x: -x[1])
    for i in range(k):
        new_label_list.append(sorted_label_list[i][0])
    return new_label_list

def loadArcMaps(path, threshold=0.02):
    def reduce_op(s):
        if len(s) < 3:
            return s
        if s[:2] == "op" and s[2] in "0123456789":
            return "op"
        return s
    arc_choices = defaultdict(set)
    with open(path, "r", encoding='utf-8') as arc_f:
        for line in arc_f:
            line = line.strip()
            if line:
                splits = line.split("\t")
                assert len(splits) == 3, "Incorrect arc map!"
                symbol = splits[0].strip()
                total_count = int(splits[1])
                arc_tuples = [(item.split(":")[0], int(item.split(":")[1])) for item
                              in splits[2].strip().split(";")]
                # total_count = sum([count for (_, count) in arc_tuples])
                ceiling = total_count * (1-threshold)
                curr_total = 0.0
                for (l, count) in arc_tuples:
                    if curr_total > ceiling:
                        break
                    curr_total += count
                    arc_choices[symbol].add(reduce_op(l))
    return arc_choices

def defaultArcChoices():
    frequent_rels = ["ARG0", "ARG1", "ARG2", "mod", "INSTANCE"]
    return set(["L-%s" % item for item in frequent_rels]) | set(["R-%s" % item for item in frequent_rels])

def loadCounter(path, delimiter="\t"):
    actions = set()
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                actions.add(line.strip().split(delimiter)[0])
    return actions

def loadDependency(path, tok_seqs, align=False, ulfdep=False):
    def dep_equal(first, second):
        first_repr = first.replace("`", "\'")
        second_repr = second.replace("`", "\'")
        return first_repr == second_repr

    def get_alignmaps():
        dep_tok_seqs = loadDepTokens(path)

        assert len(tok_seqs) == len(dep_tok_seqs)
        align_maps = []
        for (i, tok_seq) in enumerate(tok_seqs):
            dep_seq = dep_tok_seqs[i]
            align_map = {}
            tok_idx = 0
            curr_repr = ""
            for (k, dep_tok) in enumerate(dep_seq):
                curr_repr += dep_tok
                align_map[k] = tok_idx
                if dep_equal(curr_repr, tok_seq[tok_idx]):
                    tok_idx += 1
                    curr_repr = ""
            try:
                assert tok_idx == len(tok_seq)
            except:
                print(tok_seq)
                print(dep_seq)
                sys.exit(1)
            # print(align_map)
            align_maps.append(align_map)
        return align_maps

    dep_idx = 7
    if ulfdep:
      dep_idx = 6
    dep_trees = []
    align_maps = None
    if align:
        align_maps = get_alignmaps()
    with open(path, "r", encoding='utf-8') as dep_f:
        tree = DependencyTree()
        sent_idx = 0
        tok_idx = 0
        last_idx = -1
        toks = tok_seqs[sent_idx]
        # index_map = {}
        offset = 0
        prev_tok_idx = -1
        align_map = None
        if align:
            align_map = align_maps[sent_idx]
        for line in dep_f:
            splits = line.strip().split("\t")
            if len(splits) < dep_idx + 1:
                tree.buildDepDist(tree.dist_threshold)
                dep_trees.append(tree)
                # print(toks)
                #print(ulfdep)
                #print(dep_idx)
                #print(splits)
                assert len(toks) == len(tree.head_list), "%s %s %d %d" % (str(toks), str(tree.head_list), len(toks), len(tree.head_list))

                tree = DependencyTree()
                sent_idx += 1
                tok_idx = 0
                offset = 0
                prev_tok_idx = -1
                if sent_idx < len(tok_seqs):
                    toks = tok_seqs[sent_idx]
                    if align:
                        align_map = align_maps[sent_idx]
                    last_idx = -1
            else:
                #print(splits)
                dep_label = splits[dep_idx]

                tok_idx = int(splits[0]) - 1
                if tok_idx <= prev_tok_idx:
                    assert tok_idx == 0 # start a new sentence in dependency.
                    offset += (prev_tok_idx+1)
                prev_tok_idx = tok_idx

                if align:
                    tok_idx = align_map[tok_idx]
                    if tok_idx == last_idx:
                        continue
                    last_idx = tok_idx
                head_idx = int(splits[dep_idx - 1]) - 1 # root becomes -1.
                if align and head_idx >= 0:
                    head_idx = align_map[head_idx]
                if head_idx != -1:
                    head_idx += offset

                tree.add(head_idx, dep_label)
        dep_f.close()
    return dep_trees

def loadULFAMRConll(path):
    ulfamr_graphs = []
    with open(path, "r", encoding='utf-8') as amr_f:
        graph = ULFAMRGraph()
        visited = set()
        root_num = 0
        sent_idx = 0
        for line in amr_f:
            splits = line.strip().split("\t")
            if not line.strip():
                # In between AMRs, so save and clear variables.
                graph.buildEdgeMap()

                # TODO: change the symbol to word mapping.
                graph.buildWordToSymbolIDX()
                ulfamr_graphs.append(graph)
                try:
                    assert root_num == 1, "Sentence %d" % sent_idx
                except:
                    print("cycle at sentence %d" % sent_idx)

                graph = ULFAMRGraph()
                root_num = 0
            elif splits[0].startswith("sentence "):  # sentence index
                # Sentence index label.
                visited = set()
                sent_idx += 1
                # print("Sentence %d" % sent_idx)
            else:
                # Parse graph line.
                if len(splits) != 8:
                    print("Length inconsistent in conll format %s" % len(splits), file=sys.stderr)
                    print(" ".join(splits), file=sys.stderr)
                    sys.exit(1)
                #print("splits {}".format(splits))
                symbol_idx = int(splits[0])
                is_var = bool(splits[1])
                symbol_label = splits[2]
                # TODO: replace this with a span, and make further changes.
                word_idx = splits[3]
                outgoing_rels = splits[4]
                parent_rels = splits[5]
                _ = splits[6] # Old, category label
                map_label = splits[7]
                s = SymbolLabel(symbol_label)
                s.setVarType(is_var)
                s.map_info = map_label
                if word_idx == "NONE":
                    s.aligned = False
                else:
                    s.aligned = True
                    # wids = word_idx.split("#")
                    #print("word_idx {}".format(word_idx))
                    word_span = word_idx.split("-")
                    start, end = int(word_span[0]), int(word_span[1])
                    curr_set = set(range(start, end))
                    # TODO: there can be repeated spans.
                    # assert len(visited & curr_set) == 0, "\nVisited span: %d-%d\n" % (start, end)
                    if len(visited & curr_set) != 0:
                        s.aligned = False
                    else:
                        assert end > start
                        visited |= curr_set

                        # Assume we only align symbol to the last position of a span.
                        s.addAlignment(end-1)
                        s.aligned = True
                        s.setSpan((start, end))
                    # align = False
                    # TODO: change here to have alignments.
                    # for wid in wids:
                    #     w_idx = int(wid)
                    #     if w_idx not in visited:
                    #         s.addAlignment(w_idx)
                    #         align = True
                    #         visited.add(w_idx)
                    #         break
                    # if not align:
                    #     s.aligned = False

                # Processing outgoing relations.
                if outgoing_rels != "NONE":
                    out_rels = outgoing_rels.split("#")
                    for rel in out_rels:
                        fields = rel.split(":")
                        s.rels.append(fields[0])
                        s.tail_ids.append(int(fields[1]))

                # Processing incoming relations
                if parent_rels == "NONE":
                    graph.setRoot(symbol_idx)
                    root_num += 1
                else:
                    in_rels = parent_rels.split("#")
                    for rel in in_rels:
                        fields = rel.split(":")
                        s.parent_rels.append(fields[0])
                        s.parent_ids.append(int(fields[1]))

                s.buildRelMap()
                graph.addSymbol(s)
    return ulfamr_graphs

def loadDataset(path, ulfdep=True):
    def ignore_instances(seqs, ignored_set):
        tok_seqs = [seq for (idx, seq) in enumerate(seqs) if idx not in ignored_set]
        return tok_seqs

    tok_file = os.path.join(path, "token")
    lemma_file = os.path.join(path, "lemma")
    pos_file = os.path.join(path, "pos")
    ner_file = os.path.join(path, "ner")
    dep_file = os.path.join(path, "dep")
    ulfamr_conll_file = os.path.join(path, "ulfamr_conll")

    tok_seqs = loadTokens(tok_file)
    lem_seqs = loadTokens(lemma_file)
    pos_seqs = loadTokens(pos_file)
    ner_seqs = loadTokens(ner_file) # NB: the original NER annotations are aligned, but prepareTokens simplifies them into a typical token format.
    dep_trees = loadDependency(dep_file, tok_seqs, ulfdep=ulfdep)
    ulfamr_graphs = loadULFAMRConll(ulfamr_conll_file)

    assert len(tok_seqs) == len(ulfamr_graphs), \
            "len(tok_seqs): {}\tlen(ulfamr_graphs): {}".format(len(tok_seqs), len(ulfamr_graphs))

    dataset = Dataset(top_arc=50)
    dataset.setTok(tok_seqs)
    dataset.setLemma(lem_seqs)
    dataset.setPOS(pos_seqs)
    dataset.setNER(ner_seqs)
    dataset.setDepTrees(dep_trees)
    dataset.setULFAMRGraphs(ulfamr_graphs)

    return dataset


def saveCounter(counts, path):
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    with open(path, "w") as wf:
        for (item ,count) in sorted_items:
            print("%s\t%d" % (item, count), file=wf)
        wf.close()


#!/usr/bin/python3
import random
from amr_utils import *
import logger
import argparse
import alignment_utils
from entities import identify_entities
from constants import *
from date_extraction import *
from utils import *
import ml_utils
from oracle_data import *
import sys
from exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()

def collapseTokens(tok_seq, lemma_seq, pos_seq, ner_seq, span_to_type, isTrain=True, span_max=6):
    n_toks = len(tok_seq)
    collapsed = set()

    new_alignment = defaultdict(list)
    collapsed_seq = []
    collapsed_lem = []
    collapsed_pos = []
    collapsed_ner = []
    start_to_end = {}
    # print "Original tokens:", " ".join(tok_seq)
    # tok_seq just separate a sentence by space
    for i in range(n_toks):    # loop over all tokens
        if i in collapsed:  # Not dealing with collapsed (Empty at first)
            continue
        for j in range(i+span_max, i, -1): # only deal with 6 beforehand
        #for j in range(i+1, n_toks+1):
            # using the span of a syntax to get the type
            if (i, j) in span_to_type:  # span_to_type like a disctionary, collecting vector of size 3, (index, symatic, symatic)
                node_idx, _, curr_sym = span_to_type[(i, j)]
                aligned_set = set(range(i, j)) # Collection of numbers?
                # collapsed & aligned_set means intersection of the two
                if len(collapsed & aligned_set) != 0:   # skip non-trivial intersection case
                    continue
                collapsed |= aligned_set    # collapsed = collapsed | aligned_set (union)
                curr_idx = len(collapsed_seq)   # skip those have already been collapsed
                # pair alignment of nodes with current index in sequence?
                new_alignment[node_idx].append((curr_idx, curr_idx+1))
                # An array that tells the end position given input index
                start_to_end[i] = j
                if 'NE_' in curr_sym or 'DATE' in curr_sym or 'NUMBER' in curr_sym or curr_sym == "NE":
                    if "NE_" in curr_sym:   # curr_sym current symatic
                        rd = random.random()
                        if rd >= 0.9 and isTrain:
                            curr_sym = "NE" # What does NE represents
                    collapsed_seq.append(curr_sym)
                    collapsed_lem.append(curr_sym)
                    if 'NE' in curr_sym:
                        collapsed_pos.append('NE')
                    elif 'DATE' in curr_sym:
                        collapsed_pos.append('DATE')
                    else:
                        collapsed_pos.append('NUMBER')
                    collapsed_ner.append('O')
                # This portion unchecked yet
                elif j - i > 1:
                    # print "Categorized phrase:", tok_seq[i:j]
                    # Don't actually collapse these.
                    for k in range(i, j):
                      collapsed_seq.append(tok_seq[k])
                      collapsed_lem.append(lemma_seq[k].lower())
                      collapsed_pos.append(pos_seq[k])
                      collapsed_ner.append(ner_seq[k])
                    del start_to_end[i] # Remove this from start to end since we're not actually collapsing this.
                # This portion unchecked yet
                else:
                    collapsed_seq.append(tok_seq[j-1])
                    collapsed_lem.append(lemma_seq[j-1].lower())
                    collapsed_pos.append(pos_seq[j-1])
                    collapsed_ner.append(ner_seq[j-1])
        # Uncheck yet. Need input to contains brackets?
        if i not in collapsed:
            # assert i not in collapsed
            if "LRB" in tok_seq[i]:
                tok_seq[i] = '('
                lemma_seq[i] = '('
            elif "RRB" in tok_seq[i]:
                tok_seq[i] = ')'
                lemma_seq[i] = ')'
            collapsed_seq.append(tok_seq[i])
            collapsed_lem.append(lemma_seq[i].lower())
            collapsed_pos.append(pos_seq[i])
            collapsed_ner.append(ner_seq[i])
            collapsed.add(i)
    try:
        assert len(collapsed) == len(tok_seq)   # if finish
    except:
        print("Everything should be collapsed in the new sequence: %s." % str(span_to_type))
        noncollapsed = [(k, tok) for (k, tok) in enumerate(tok_seq) if k not in collapsed]
        for k, tok in noncollapsed:
            print("Not collapsed %d : %s" % (k, tok))
        sys.exit(1)
    return collapsed_seq, collapsed_lem, collapsed_pos, collapsed_ner, new_alignment, start_to_end

#For the unaligned concepts in the AMR graph
def unalignedOutTails(amr, all_alignments):
    tail_set = defaultdict(set)
    index_set = set()
    stack = [amr.root]
    visited = set()

    edge_list = []
    # connects child and parent, dictionary of set, use node index as searching index
    edge_map = defaultdict(set)

    # Current amr tree and ulf tree exactly the same
    # Loop over whenever stack is non-empty
    while stack:
        # Stack stores the index of the node
        curr_node_idx = stack.pop()

        # Name of the node and name of the variable
        curr_node = amr.nodes[curr_node_idx]
        curr_var = curr_node.node_str()

        if curr_node_idx in visited: #A reentrancy found
            continue

        index_set.add(curr_node_idx)
        visited.add(curr_node_idx)
        unaligned = curr_node_idx not in all_alignments

        # Why need reverse?
        for edge_idx in reversed(curr_node.v_edges):
            curr_edge = amr.edges[edge_idx]
            child_idx = curr_edge.tail
            stack.append(child_idx)
            # if (unaligned) and (child_idex != curr_node_idx)
            if unaligned and child_idx != curr_node_idx: #Avoid self edge
                # tail_set contains all child that are not aligned
                tail_set[curr_node_idx].add(child_idx)

            edge_map[curr_node_idx].add(child_idx)
            edge_map[child_idx].add(curr_node_idx)
            edge_list.append((curr_node_idx, child_idx))

    return index_set, tail_set, edge_map, edge_list

def visitUnaligned(mapped_set, amr, index):
    assert index in mapped_set
    stack = [index]
    visited_seq = []
    visited = set()
    while stack:
        # to_remove = set()
        curr_idx = stack.pop()
        if curr_idx in visited:
            continue
        visited.add(curr_idx)

        # delete object as we have already visited it
        del mapped_set[curr_idx]
        # just the order one visits the stack
        visited_seq.append(curr_idx)

        curr_node = amr.nodes[index]
        for edge_idx in curr_node.p_edges:
            # Edge type are args, complex, instance edge(?)
            parent_edge = amr.edges[edge_idx]
            head_idx = parent_edge.head
            if head_idx in mapped_set:
                # Append all its parents
                stack.append(head_idx)
    return mapped_set, visited_seq


def buildPiSeq(amr, tok_seq, all_alignments, sorted_idxes):

    index_set, tail_set, edge_map, edge_list = unalignedOutTails(amr, all_alignments)

    pi_seq = []
    visited = set()
    """
    for index in sorted_idxes:
        if index in visited:
            continue

        visited.add(index)
        #pi_seq.append(index)
        curr_node = amr.nodes[index]
        for edge_idx in curr_node.p_edges:
            parent_edge = amr.edges[edge_idx]
            head_idx = parent_edge.head
            if head_idx in tail_set:
                assert index in tail_set[head_idx]

                tail_set, leaf_seq = visitUnaligned(tail_set, amr, head_idx)
                if leaf_seq:
                    pi_seq.extend(leaf_seq)
                    visited |= set(leaf_seq)
        pi_seq.append(index)

    for index in index_set:
        if index not in visited:
            pi_seq.append(index)
    """

    helper(amr, amr.root, pi_seq)

    assert len(pi_seq) == len(index_set)

    return pi_seq, edge_map, edge_list

# dfs for the tree
def helper(amr, node_idx, pi_seq):
    curr_node = amr.nodes[node_idx]

    pi_seq.append(node_idx)
    for edge_idx in curr_node.v_edges:
        curr_edge = amr.edges[edge_idx]
        child_idx = curr_edge.tail
        helper(amr, child_idx, pi_seq)


class AMR_stats(object):
    def __init__(self):
        self.num_reentrancy = 0
        self.num_predicates = defaultdict(int)
        self.num_nonpredicate_vals = defaultdict(int)
        self.num_consts = defaultdict(int)
        self.num_named_entities = defaultdict(int)
        self.num_entities = defaultdict(int)
        self.num_relations = defaultdict(int)

    def update(self, local_re, local_pre, local_non, local_con, local_ent, local_ne):
        self.num_reentrancy += local_re
        for s in local_pre:
            self.num_predicates[s] += local_pre[s]

        for s in local_non:
            self.num_nonpredicate_vals[s] += local_non[s]

        for s in local_con:
            self.num_consts[s] += local_con[s]

        for s in local_ent:
            self.num_entities[s] += local_ent[s]

        for s in local_ne:
            self.num_named_entities[s] += local_ne[s]
        #for s in local_rel:
        #    self.num_relations[s] += local_rel[s]

    def collect_stats(self, amr_graphs):
        for amr in amr_graphs:
            (named_entity_nums, entity_nums, predicate_nums, variable_nums, const_nums, reentrancy_nums) = amr.statistics()
            self.update(reentrancy_nums, predicate_nums, variable_nums, const_nums, entity_nums, named_entity_nums)

    def dump2dir(self, dir):
        def dump_file(f, dict):
            sorted_dict = sorted(dict.items(), key=lambda k:(-k[1], k[0]))
            for (item, count) in sorted_dict:
                print('%s %d' % (item, count), file=f)
            f.close()

        pred_f = open(os.path.join(dir, 'pred'), 'w')
        non_pred_f = open(os.path.join(dir, 'non_pred_val'), 'w')
        const_f = open(os.path.join(dir, 'const'), 'w')
        entity_f = open(os.path.join(dir, 'entities'), 'w')
        named_entity_f = open(os.path.join(dir, 'named_entities'), 'w')
        #relation_f = open(os.path.join(dir, 'relations'), 'w')

        dump_file(pred_f, self.num_predicates)
        dump_file(non_pred_f, self.num_nonpredicate_vals)
        dump_file(const_f, self.num_consts)
        dump_file(entity_f, self.num_entities)
        dump_file(named_entity_f, self.num_named_entities)
        #dump_file(relation_f, self.num_relations)

    def loadFromDir(self, dir):
        def load_file(f, dict):
            for line in f:
                item = line.strip().split(' ')[0]
                count = int(line.strip().split(' ')[1])
                dict[item] = count
            f.close()

        pred_f = open(os.path.join(dir, 'pred'), 'r')
        non_pred_f = open(os.path.join(dir, 'non_pred_val'), 'r')
        const_f = open(os.path.join(dir, 'const'), 'r')
        entity_f = open(os.path.join(dir, 'entities'), 'r')
        named_entity_f = open(os.path.join(dir, 'named_entities'), 'r')

        load_file(pred_f, self.num_predicates)
        load_file(non_pred_f, self.num_nonpredicate_vals)
        load_file(const_f, self.num_consts)
        load_file(entity_f, self.num_entities)
        load_file(named_entity_f, self.num_named_entities)

    def __str__(self):
        s = ''
        s += 'Total number of reentrancies: %d\n' % self.num_reentrancy
        s += 'Total number of predicates: %d\n' % len(self.num_predicates)
        s += 'Total number of non predicates variables: %d\n' % len(self.num_nonpredicate_vals)
        s += 'Total number of constants: %d\n' % len(self.num_consts)
        s += 'Total number of entities: %d\n' % len(self.num_entities)
        s += 'Total number of named entities: %d\n' % len(self.num_named_entities)

        return s


def realign(input_file, tokenized_file, alignment_file, alignment_output):
    def ordered_alignment(align_seq):
        align_map = defaultdict(set)
        for curr_align in align_seq:
            tok_idx = int(curr_align.split("-")[0])
            graph_idx = curr_align.split("-")[1]
            align_map[tok_idx].add(graph_idx)
        sorted_align = sorted(align_map.items(), key=lambda x: x[0])
        return sorted_align

    orig_seqs = loadTokens(input_file)
    tokenized_seqs = loadTokens(tokenized_file)
    alignments = loadTokens(alignment_file)

    with open(alignment_output, 'w') as wf:
        for (sent_idx, orig_seq) in enumerate(orig_seqs):
            tokenized_seq = tokenized_seqs[sent_idx]
            alignment = alignments[sent_idx]
            new_alignment = []
            new_idx = 0

            sorted_align = ordered_alignment(alignment)
            for orig_idx, align_set in sorted_align:
                orig_tok = orig_seq[orig_idx]
                (new_start, new_end) = searchSeq(orig_tok, tokenized_seq, new_idx)
                if new_start == -1:
                    print("Something wrong with sentence %d" % sent_idx)
                    print(orig_seq)
                    print(tokenized_seq)
                    print(orig_idx, orig_tok)
                    print(new_idx, tokenized_seq[new_idx])
                    sys.exit(1)
                elif "".join(tokenized_seq[new_start:new_end]) != orig_tok:
                    print("Changes made from #%s# to #%s#" % (orig_tok, " ".join(tokenized_seq[new_start:new_end])))

                if new_end - new_start > 1:
                    print("Align to multiple: #%s# to #%s#" % (orig_tok, " ".join(tokenized_seq[new_start:new_end])))

                for idx in range(new_start, new_end):
                    for curr_align in align_set:
                        new_align = "%d-%s" % (idx, curr_align)
                        new_alignment.append(new_align)
                new_idx = new_end
            print(" ".join(new_alignment), file=wf)
        wf.close()

def dependency_align(dep_file, token_file, output_dep_file=None):
    def seq_equal(first, second):
        first_repr = first.replace("`", "\'")
        second_repr = second.replace("`", "\'")
        return first_repr == second_repr
    tok_seqs = loadTokens(token_file)
    dep_tok_seqs = loadDepTokens(dep_file)

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
            if seq_equal(curr_repr, tok_seq[tok_idx]):
                tok_idx += 1
                curr_repr = ""
        assert tok_idx == len(tok_seq)
        align_maps.append(align_map)
    return align_maps


def linearize_amr(args):

    # Appends token features of the current type into the feats list.
    def tok_features(feats, seq, curr_idx, win=2, prefix=""):
        length = len(seq)
        for idx in range(curr_idx-win, curr_idx+win+1):
            feat_str = "%s%d=" % (prefix, idx-curr_idx)
            if idx < 0 or idx >= length:
                feats.append(feat_str+ "-NULL-")
            else:
                tok_str = seq[idx]
                if tok_str[:3] == "NE_":
                    tok_str = "NE"
                feats.append(feat_str+ tok_str)

    os.system("mkdir -p %s" % args.run_dir)
    logger.file = open(os.path.join(args.run_dir, 'logger'), 'w')

    phrase_map = {}

    amr_file = os.path.join(args.data_dir, 'amr')
    # The initial alignment files (one per line and as specified by the AMR project)
    alignment_file = os.path.join(args.data_dir, 'alignment.amr')
    tok_file = os.path.join(args.data_dir, 'token')
    lem_file = os.path.join(args.data_dir, 'lemma')
    pos_file = os.path.join(args.data_dir, 'pos')
    ner_file = os.path.join(args.data_dir, 'ner')

    amr_graphs = load_amr_graphs(amr_file, ignore_firstnlast=False)
    alignments = loadTokens(alignment_file)
    toks = loadTokens(tok_file)
    lems = loadTokens(lem_file)
    poss = loadTokens(pos_file)
    ners = loadNERTokens(ner_file)

    multi_map = defaultdict(int)

    assert len(amr_graphs) == len(alignments) and len(amr_graphs) == len(toks) and len(amr_graphs) == len(poss)and len(amr_graphs) == len(ners),  '%d %d %d %d %d' % (len(amr_graphs), len(alignments), len(toks), len(poss), len(ners))

    amr_statistics = AMR_stats()

    if args.use_stats:
        amr_statistics.loadFromDir(args.stats_dir)
    else:
        os.system('mkdir -p %s' % args.stats_dir)
        amr_statistics.collect_stats(amr_graphs)
        amr_statistics.dump2dir(args.stats_dir)

    # phrases = set([line.strip().split('###')[0].strip() for line in open('phrases')])
    # phrase_freq = defaultdict(int)

    conll_wf = open(args.conll_file, 'w')
    output_tok = os.path.join(args.run_dir, "token")
    output_lem = os.path.join(args.run_dir, "lemma")
    output_pos = os.path.join(args.run_dir, "pos")
    output_ner = os.path.join(args.run_dir, "ner")

    output_alignment = os.path.join(args.run_dir, "alignment")
    old_align_output = os.path.join(args.run_dir, "old_align")
    concept_alignment = os.path.join(args.run_dir, "concept_alignment")
    conceptID_file = os.path.join(args.run_dir, "conceptID_examples.txt")

    alignment_counts = {}
    lemma_align_counts = {}
    tok_counts = defaultdict(int)
    lem_counts = defaultdict(int)

    tok_wf = open(output_tok, 'w')
    lemma_wf = open(output_lem, 'w')
    pos_wf = open(output_pos, 'w')
    ner_wf = open(output_ner, 'w')
    alignment_wf = open(output_alignment, 'w')
    old_align_wf = open(old_align_output, 'w')
    concept_example_wf = open(conceptID_file, 'w')

    root_entity_num = defaultdict(int)

    mle_map = {}
    mleLemmaMap = {}

    tok2counts = {}
    tok2counts["NE"] = defaultdict(int)
    tok2counts["NUMBER"] = defaultdict(int)
    tok2counts["DATE"] = defaultdict(int)
    lemma2counts = {}

    random.seed(0)

    conceptToOutGo = {}
    conceptToIncome = {}

    conceptOutgoFreq = defaultdict(int)
    conceptIncomeFreq = defaultdict(int)

    # phrase_counts = loadCountTable("phrase_dist.txt")
    #ambiguous_toks = ml_utils.loadConceptIDCounts("train_tables/conceptIDcounts.txt")
    #ambiguous_lems = ml_utils.loadConceptIDCounts("train_tables/lemmaIDcounts.txt")
    ambiguous_toks = []
    ambiguous_lems = []


    concept_counts = defaultdict(int)
    relcounts = defaultdict(int)

    #frequent_set = loadFrequentSet(args.freq_dir)
    frequent_set = set()

    unaligned_sents = 0

    tok_to_categories = {}
    tok_to_concept = {}

    #ml_utils.load_mle("./train_tables/tokToConcepts.txt", tok_to_concept)
    #ml_utils.load_mle("./train_tables/lemToConcepts.txt", tok_to_concept)

    for tok_str in tok_to_concept:
        mapped_category = getCategories(tok_to_concept[tok_str], frequent_set)
        tok_to_categories[tok_str] = mapped_category

    ngram_counts = defaultdict(int)

    #print "Number of toks entries: {}".format(len(toks))
    for (sent_idx, tok_seq) in enumerate(toks):
        #print "sent_idx {}\ntok_seq {}".format(sent_idx, tok_seq)
        lemma_seq, pos_seq, ner_seq = lems[sent_idx], poss[sent_idx], ners[sent_idx]
        alignment_seq, amr = alignments[sent_idx], amr_graphs[sent_idx]
        #print "alignment_seq: {}".format(alignment_seq)

        tok2rels = defaultdict(set)

        logger.writeln('Sentence #%d' % (sent_idx+1))
        logger.writeln(' '.join(tok_seq))

        amr.setStats(amr_statistics)

        amr.set_sentence(lemma_seq) ##Here we consider the lemmas for amr graph
        amr.set_poss(pos_seq)
        amr.set_ners(ner_seq)


        # Initialize the alignment
        node_alignment, _ = alignment_utils.initializeAlignment(amr)

        concept_align_map = defaultdict(list)
        relation_align_map = defaultdict(list)

        node_to_toks, edge_to_toks, temp_aligned = alignment_utils.extractNodeMapping(
            alignment_seq, amr, concept_align_map, relation_align_map)
        alignment_utils.outputEdgeAlignment(tok_seq, amr, edge_to_toks, tok2rels)

        temp_unaligned = set(range(len(pos_seq))) - temp_aligned

        aligned_toks = set()

        all_alignments = defaultdict(list)
        nodeid_to_frag = {}

        entity_toks = set()

        entity_not_align = alignment_utils.alignEntities(tok_seq, amr, alignment_seq, nodeid_to_frag, entity_toks,
                                                         aligned_toks, all_alignments, temp_unaligned, node_alignment)

        #Verbalization list
        verb_map = defaultdict(set)
        alignment_utils.alignVerbalization(tok_seq, lemma_seq, amr, verb_map, all_alignments, verb_map,
                                           aligned_toks, node_alignment, multi_map)

        aligned_nodes = set([node_idx for (node_idx, aligned) in enumerate(node_alignment) if aligned])

        alignment_utils.alignOtherConcepts(tok_seq, lemma_seq, amr, aligned_toks, aligned_nodes, node_to_toks,
                                           edge_to_toks, all_alignments, multi_map)
        # print "All alignments:", str(all_alignments)

        ##Based on the alignment from node index to spans in the string
        unaligned_set = set(range(len(pos_seq))) - aligned_toks
        unaligned_idxs = sorted(list(unaligned_set))
        logger.writeln("Unaligned tokens: %s" % (" ".join([tok_seq[i] for i in unaligned_idxs])))

        unaligned_nodes = amr.unaligned_nodes(aligned_nodes)
        logger.writeln("Unaligned vertices: %s" % " ".join([node.node_str() for node in unaligned_nodes]))

        # Save alignments to file.
        start2end, category_map, node_map, wiki_map = {}, {}, {}, {}
        for node_idx, span_list in all_alignments.items():
            for span in span_list:
                if span[1] - span[0] > 6:
                    continue
                start2end[span[0]] = span[1]
                wiki_map[span[0]] = span[2]
                category_map[span[0]] = span[3]
                node_map[span[0]] = amr.nodes[node_idx].node_str()

        # Save the old alignment.
        print("Sentence #%d" % sent_idx, file=old_align_wf)
        for (tok_idx, curr_tok) in enumerate(tok_seq):
            concept_l = "NONE"
            if tok_idx in concept_align_map:
                concept_l = "#".join(concept_align_map[tok_idx])
            relation_l = "NONE"
            if tok_idx in relation_align_map:
                relation_l = "#".join(relation_align_map[tok_idx])
            attrs = ["%d" % tok_idx]
            attrs.append(curr_tok)
            attrs.append(concept_l)
            attrs.append(relation_l)
            print(" ||| ".join(attrs), file=old_align_wf)
        print("", file=old_align_wf)

        visited_idxs = set()

        conceptID_seq = []

        if entity_not_align:
            unaligned_sents += 1
            logger.writeln(str(amr))

        new_amr, _, span_to_type = AMRGraph.collapsedAMR(amr, all_alignments)

        new_amr.update_stats(conceptToOutGo, conceptToIncome, conceptOutgoFreq,
                             conceptIncomeFreq, concept_counts, relcounts, frequent_set)

        nodeid_to_repr = {}
        for (start, end) in span_to_type:
            (node_idx, subgraph_repr, category) = span_to_type[(start, end)]
            if node_idx not in nodeid_to_repr:
                tok_repr = "_".join(tok_seq[start:end])
                nodeid_to_repr[node_idx] = (tok_repr, subgraph_repr, category)

            tok_s = ' '.join(tok_seq[start:end])
            lemma_s = ' '.join(lemma_seq[start:end])
            if not tok_s in tok2counts:
                tok2counts[tok_s] = defaultdict(int)

            tok2counts[tok_s]["%s||%s" % (category, subgraph_repr)] += 1
            tok_counts[tok_s] += 1
            if category == "NE" or "NE_" in category:

                tok2counts["NE"]["%s||%s" % (category, "-")] += 1
                tok_counts["NE"] += 1
            elif category == "DATE" or category == "NUMBER":
                tok2counts[category]["%s||%s" % (category, "-")] += 1
                tok_counts[category] += 1
            if not ((category == "NE") or ("NE_" in category) or (category == "DATE")):
                if not lemma_s in lemma2counts:
                    lemma2counts[lemma_s] = defaultdict(int)

                lemma2counts[lemma_s]["%s||%s" % (category, subgraph_repr)] += 1
                lem_counts[lemma_s] += 1

        for index in unaligned_idxs:
            tok_s = tok_seq[index]
            lemma_s = lemma_seq[index]
            if not tok_s in tok2counts:
                tok2counts[tok_s] = defaultdict(int)
            if not lemma_s in lemma2counts:
                lemma2counts[lemma_s] = defaultdict(int)
            tok2counts[tok_s]["%s||%s" % ("NONE", "NONE")] += 1
            tok_counts[tok_s] += 1
            lemma2counts[lemma_s]["%s||%s" % ("NONE", "NONE")] += 1
            lem_counts[lemma_s] += 1

        # print "Span to type:", str(span_to_type)
        (
            collapsed_toks,
            collapsed_lem,
            collapsed_pos,
            collapsed_ner,
            new_alignment,
            new_start_to_end
        ) = collapseTokens(tok_seq, lemma_seq, pos_seq, ner_seq, span_to_type, True)

        # Save the new alignments.
        print("Sentence #%d" % sent_idx, file=alignment_wf)
        for (tok_idx, curr_tok) in enumerate(tok_seq):

            if tok_idx in visited_idxs:
                continue
            start = tok_idx
            rel_align_str = "NONE"
            if tok_idx in tok2rels:
                rel_align_str = "#".join([amr.edges[edge_idx].label for edge_idx in tok2rels[tok_idx]])

            if start in start2end and start in new_start_to_end:
                end, wiki_l, cate_l, node_l = new_start_to_end[start], wiki_map[start], category_map[start], node_map[start]
                if cate_l == "MULT":
                    root_entity_num[node_l] += 1

                visited_idxs |= set(range(start, end))
                assert node_l != "NONE", "%d-%d, %s, %s\n" % (start, end, node_l, cate_l)
            else:
                end, wiki_l, cate_l, node_l = start + 1, "NONE", "NONE", "NONE"
                visited_idxs.add(start)

            attrs = ["%d-%d" % (start, end)]
            tok_str = " ".join(tok_seq[start:end])
            lem_str = " ".join(lemma_seq[start:end])
            attrs.append(tok_str)
            attrs.append(cate_l)
            attrs.append(node_l)
            concept_category = getCategories(node_l, frequent_set)
            attrs.append(concept_category)
            attrs.append(rel_align_str)
            attrs.append(wiki_l)

            concept_str = ""
            if tok_str in tok_to_categories:
                concept_str = tok_to_concept[tok_str]
                concept_category = tok_to_categories[tok_str]
            elif lem_str in tok_to_categories:
                concept_str = tok_to_concept[lem_str]
                concept_category = tok_to_categories[lem_str]
            conceptID_seq.append((tok_str, lem_str, in_bracket(start, end, tok_seq), wiki_l,
                                  concept_str, concept_category))

            print(" ||| ".join(attrs), file=alignment_wf)
        print("", file=alignment_wf)

        assert len(conceptID_seq) == len(collapsed_toks), "%s\n%s\n" % (str(conceptID_seq), str(collapsed_toks))

        # Save the concept alignment
        mle_concept_seq = [concept for (_, _, _, _, concept, _) in conceptID_seq]
        mle_category_seq = [category for (_, _, _, _, _, category) in conceptID_seq]
        for (tok_idx, curr_tok) in enumerate(collapsed_toks):
            # Build feature vector.
            curr_tok_str, curr_lem_str, bracket, wiki_repr, _, category = conceptID_seq[tok_idx]
            if curr_tok == "NE" or "NE_" in curr_tok or curr_tok == "DATE" or curr_tok == "NUMBER":
                continue

            else:
                if curr_tok_str in ambiguous_toks or curr_lem_str in ambiguous_lems:
                    feats = [wiki_repr]
                else:
                    continue
            tok_features(feats, collapsed_toks, tok_idx, 2, "tok")
            tok_features(feats, collapsed_lem, tok_idx, 2, "lem")
            tok_features(feats, collapsed_pos, tok_idx, 3, "pos")
            tok_features(feats, collapsed_ner, tok_idx, 3, "ner")
            tok_features(feats, mle_concept_seq, tok_idx, 2, "concept")
            tok_features(feats, mle_category_seq, tok_idx, 3, "cate")
            feats.append("tok=%s" % curr_tok_str)
            feats.append("lem=%s" % curr_lem_str)
            if bracket:
                feats.append("inbracket=true")
            else:
                feats.append("inbracket=false")

            print("\t".join(feats), file=concept_example_wf)

        assert len(tok_seq) == len(pos_seq)
        assert len(tok_seq) == len(ner_seq)

        new_amr.set_sentence(collapsed_toks)
        new_amr.set_lemmas(collapsed_lem)
        new_amr.set_poss(collapsed_pos)
        new_amr.set_ners(collapsed_ner)
        new_amr.setStats(amr_statistics)

        span_idxes = []
        for node_idx in new_alignment:
            for (start, end) in new_alignment[node_idx]:
                span_idxes.append((start, end, node_idx))
        sorted_idxes = sorted(span_idxes, key=lambda x: (x[0], x[1]))

        sorted_idxes = [z for (x, y, z) in sorted_idxes]

        pi_seq, edge_map, edge_list = buildPiSeq(new_amr, collapsed_toks, new_alignment, sorted_idxes)
        assert len(pi_seq) == len(new_amr.nodes)

        print('sentence %d' % sent_idx, file=conll_wf)
        print(" ".join(collapsed_toks), file=tok_wf)
        print(" ".join(collapsed_lem), file=lemma_wf)
        print(" ".join(collapsed_pos), file=pos_wf)
        print(" ".join(collapsed_ner), file=ner_wf)
        #print(str(new_amr))

        origToNew = {}
        for (i, index) in enumerate(pi_seq):
            assert index not in origToNew
            origToNew[index] = i

        #Output the graph file in a conll format
        for (i, index) in enumerate(pi_seq):
            line_reps = []
            curr_node = new_amr.nodes[index]
            line_reps.append(str(i))
            word_indices = []

            subgraph_repr = "NONE"
            if index in new_alignment:
                assert index in nodeid_to_repr
                tok_repr, subgraph_repr, category = nodeid_to_repr[index]
                subgraph_repr = "%s||%s" % (tok_repr, subgraph_repr)

                for (start, end) in new_alignment[index]:
                    for tok_id in range(start, end + 1):
                        word_indices.append(tok_id)

            var_bit = '1' if curr_node.is_var_node() else '0'
            line_reps.append(var_bit)
            concept_repr = new_amr.nodes[index].node_str()
            concept_category = getCategories(concept_repr, frequent_set, True)
            # Turn spaces into underscores, unless this is a string representation.
            if " " in concept_repr and not (len(concept_repr) > 1 and concept_repr[0] == "\"" and concept_repr[-1] == "\""):
                concept_repr = "_".join(concept_repr.split())
            line_reps.append(concept_repr)
            #word_repr = '#'.join([str(tok_id) for tok_id in word_indices]) if word_indices else 'NONE'
            word_repr = 'NONE'
            if len(word_indices) > 1:
              word_repr = '{}-{}'.format(min(word_indices),max(word_indices))
            elif len(word_indices) == 1:
              word_repr = '{}-{}'.format(word_indices[0],word_indices[0]+1)
            line_reps.append(word_repr)
            child_triples = curr_node.childTriples()
            child_repr = '#'.join(['%s:%d' % (label, origToNew[tail_idx]) for (label, tail_idx) \
                    in child_triples]) if child_triples else 'NONE'
            line_reps.append(child_repr)
            parent_triples = curr_node.parentTriples()
            parent_repr = '#'.join(['%s:%d' % (label, origToNew[head_idx]) for (label, head_idx) \
                    in parent_triples]) if parent_triples else 'NONE'
            line_reps.append(parent_repr)
            line_reps.append(concept_category)
            line_reps.append(subgraph_repr)
            print('\t'.join(line_reps), file=conll_wf)
        print('', file=conll_wf)

    logger.writeln("A total of %d sentences with unaligned entities" % unaligned_sents)

    conll_wf.close()
    tok_wf.close()
    lemma_wf.close()
    pos_wf.close()
    ner_wf.close()
    alignment_wf.close()
    old_align_wf.close()
    concept_example_wf.close()

    for tok_s in tok2counts:
        sorted_map_counts = sorted(tok2counts[tok_s].items(), key=lambda x:-x[1])
        if len(sorted_map_counts) > 0 and len(sorted_map_counts[0]) > 0:
          mle_map[tok_s] = tuple(sorted_map_counts[0][0].split("||"))

    for lemma_s in lemma2counts:
        sorted_map_counts = sorted(lemma2counts[lemma_s].items(), key=lambda x:-x[1])
        mleLemmaMap[lemma_s] = tuple(sorted_map_counts[0][0].split("||"))

    # Then dump all the training statistics.
    # First edges statistics.
    if args.table_dir:

        os.system("mkdir -p %s" % args.table_dir)

        conceptIDStats = os.path.join(args.table_dir, "tokToConcepts.txt")
        lemmaConceptStats = os.path.join(args.table_dir, "lemToConcepts.txt")

        conceptOutStats = os.path.join(args.table_dir, "concept_rels.txt")
        saveMLE(conceptOutgoFreq, conceptToOutGo, conceptOutStats)

        conceptIncomeStats = os.path.join(args.table_dir, "concept_incomes.txt")
        saveMLE(conceptIncomeFreq, conceptToIncome, conceptIncomeStats)

        conceptIDcountStats = os.path.join(args.table_dir, "conceptIDcounts.txt")
        saveMLE(tok_counts, tok2counts, conceptIDcountStats)

        lemmaIDcountStats = os.path.join(args.table_dir, "lemmaIDcounts.txt")
        saveMLE(lem_counts, lemma2counts, lemmaIDcountStats)

        multigraphStats = os.path.join(args.table_dir, "multigraph_counts.txt")
        saveCounter(multi_map, multigraphStats)

        conceptStats = os.path.join(args.table_dir, "concept_counts.txt")
        saveCounter(concept_counts, conceptStats)

        relationStats = os.path.join(args.table_dir, "relation_counts.txt")
        saveCounter(relcounts, relationStats)

        multiRootStats = os.path.join(args.table_dir, "multiroots_counts.txt")
        saveCounter(root_entity_num, multiRootStats)

        categoryStats = os.path.join(args.table_dir, "train_categories.txt")
        saveSetorList(frequent_set | set(["NE", "NUMBER", "DATE", "PRED", "OTHER"]), categoryStats)

        ambiguousToksStats = os.path.join(args.table_dir, "ambiguous_toks.txt")
        ambiguousLemsStats = os.path.join(args.table_dir, "ambiguous_lems.txt")
        saveSetorList(ambiguous_toks, ambiguousToksStats)
        saveSetorList(ambiguous_lems, ambiguousLemsStats)

        #mle_map = filterNoise(mle_map, './conceptIDCounts.dict.weird.txt')
        #mleLemmaMap = filterNoise(mleLemmaMap, './lemConceptIDCounts.dict.weird.txt')

        #dumpMap(mle_map, conceptIDStats)
        #dumpMap(mleLemmaMap, lemmaConceptStats)

    # linearizeData(mle_map, mleLemmaMap, phrases, args.dev_dir, args.dev_output)
    # linearizeData(mle_map, mleLemmaMap, phrases, args.test_dir, args.test_output)

def filterNoise(curr_map, filter_file):
    weird_set = set()
    for line in open(filter_file):
        if line.strip():
            weird_set.add(line.strip().split()[0])

    new_map = {}
    for l in curr_map:
        if curr_map[l][0] == 'NONE' and l in weird_set and (not '@' in l):
            continue
        new_map[l] = curr_map[l]
    return new_map

def loadMLECount(file):
    mleCounts = {}
    count_f = open(file, 'r')
    for line in count_f:
        if line.strip():
            fields = line.strip().split(' #### ')
            word = fields[0].strip()
            choices = fields[1].split()
            for curr in choices:
                concept = curr.split(':')[0]
                count = int(curr.split(':')[1])
                if concept == '-NULL-' and count < 20:
                    print(word, count)
                else:
                    if word not in mleCounts:
                        mleCounts[word] = defaultdict(int)
                    mleCounts[word][concept] = count
    return mleCounts

def check_and_sentences(sent_str, delimiter=" ; ", tok_num=1):
    splits = sent_str.split(delimiter)
    if len(splits) > 1:
        for s in splits:
            if len(s.split()) > tok_num:
                return False
        return True
    return False

# Given the original text, generate the reformatted text. Originally, this
# function was used for AMR categorization.
def linearizeData(args, data_dir, freq_path, output_dir, save_mode=False):

    def identifyNumber(seq, mle_map):
        quantities = set(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'billion',
                          'tenth', 'million', 'thousand', 'hundred', 'viii', 'eleven', 'twelve', 'thirteen', 'iv'])
        for tok in seq:
            if tok in mle_map and mle_map[tok][0] != 'NUMBER':
                return False
            elif not (isNumber(tok) or tok in quantities):
                return False
        return True

    def replaceSymbol(s):
        return s.replace("@ - @", "@-@").replace("@ :@", "@:@").replace("@ / @", "@/@")

    def allUnaligned(seq, mleLemmaMap):
        for tok in seq:
            if not tok in mleLemmaMap:
                return False
            aligned_c = mleLemmaMap[tok][0]
            if not (aligned_c.lower() == "none"):
                return False
        return True

    os.system("mkdir -p %s" % output_dir)

    conceptIDPath = os.path.join(freq_path, "tokToConcepts.txt")
    lemmaToConceptPath = os.path.join(freq_path, "lemToConcepts.txt")
    phrasePath = os.path.join(freq_path, "phrases")

    mle_map = loadMLEFile(conceptIDPath)
    mleLemmaMap = loadMLEFile(lemmaToConceptPath)
    # phrase_map = loadPhrases(phrasePath)

    tok_file = os.path.join(data_dir, 'token')
    lem_file = os.path.join(data_dir, 'lemma')
    pos_file = os.path.join(data_dir, 'pos')
    ner_file = os.path.join(data_dir, 'ner')
    date_file = os.path.join(data_dir, 'date')

    frequent_set = loadFrequentSet(args.freq_dir)

    toks, lems, poss, ners = loadTokens(tok_file), loadTokens(lem_file), loadTokens(pos_file), loadNERTokens(ner_file)

    if save_mode:
        all_entities, entity_map = identify_entities(tok_file, ner_file, mle_map) #tok_file or lem_file?
        all_dates = dateMap(date_file)

        tokenized_file = os.path.join(data_dir, "tokenized")

        tokenized_seqs = loadTokens(tokenized_file)

        date_output = os.path.join(output_dir, "date.spans.txt")
        ner_output = os.path.join(output_dir, "ner.spans.txt")

        all_date_spans = []
        all_ner_spans = []

        abbrev_to_fullname = {}

        for (sent_idx, entities_in_sent) in enumerate(all_entities):

            tok_seq, lem_seq, pos_seq, ner_seq = toks[sent_idx], lems[sent_idx], poss[sent_idx], ners[sent_idx]
            tokenized = tokenized_seqs[sent_idx]
            date_spans = all_dates[sent_idx]

            n_toks = len(tok_seq)

            aligned_set = set()

            new_date_spans = []
            new_entity_spans = []

            aligned_new = set()

            single_dates = {}
            entity_tok_set = set()
            #Align dates
            new_idx = 0
            for (start, end) in date_spans:
                date_repr = " ".join(tok_seq[start:end])
                if date_repr in mle_map and mle_map[date_repr][0] != "DATE":
                    continue


                (new_start, new_end) = searchSeq(date_repr.replace(" ", ""), tokenized, new_idx)
                if new_start == -1:
                    print("Something wrong with sentence %d" % sent_idx)
                    print(date_repr)
                    print(tokenized)
                    sys.exit(1)
                if new_end - new_start == 1:
                    single_dates[start] = new_start
                else:
                    if new_end < len(tokenized) and tokenized[new_end] in date_suffixes:
                        new_end += 1
                        print("newly generated:", "_".join(tokenized[new_start:new_end]))
                    new_date_spans.append("%d-%d####%s" % (new_start, new_end, "_".join(tokenized[new_start:new_end])))
                    print("Date:", " ".join(tokenized[new_start:new_end]))
                    new_idx = new_end

                    aligned_set |= set(range(start, end))
                    aligned_new |= set(range(new_start, new_end))

            #First align multi tokens
            new_idx = 0
            aligned_repr = set()

            lasttok_to_repr = {}
            for (start, end, entity_typ) in entities_in_sent:
                # print 'entity:', ' '.join(tok_seq[start:end]), entity_typ
                new_aligned = set(range(start, end))
                if len(aligned_set & new_aligned) != 0:
                    continue
                abbrev = False
                ret_ent = False
                aligned_set |= new_aligned
                entity_name = ' '.join(tok_seq[start:end])
                if replaceSymbol(entity_name) in mle_map:
                    new_name = replaceSymbol(entity_name)
                    if new_name != entity_name:
                        entity_name = new_name
                        print("Replaced:", entity_name)

                if entity_name in mle_map:
                    entity_typ = mle_map[entity_name]

                    if not "NE_" in entity_typ[0] and entity_name.lower() in mle_map:
                        entity_typ = mle_map[entity_name.lower()]
                        if not "NE_" in entity_typ[0]:
                            print("Removed discovered entity:", entity_name)
                            continue
                    elif entity_typ[0] == "NONE":
                        entity_typ = ("NE", "-")
                    if entity_typ in aligned_repr:
                        print("Removed duplicate entity", entity_name)
                        entity_typ = ("REENT", "-")
                        ret_ent = True
                    aligned_repr.add(entity_typ)

                elif start == 0 and entity_name.lower() in mle_map:
                    entity_typ = mle_map[entity_name.lower()]
                    if not "NE_" in entity_typ[0]:
                        print("Removed discovered entity:", entity_name)
                        continue
                    aligned_repr.add(entity_typ)

                elif entity_typ == "PER":
                    entity_typ = ('NE_person', '-')
                else:
                    entity_typ = ('NE', '-')

                entity_repr = " ".join(tok_seq[start: end])
                if in_bracket(start, end, tok_seq) and (start - 2) in entity_tok_set:
                    try:
                        assert start - 2 in lasttok_to_repr
                        abbrev_to_fullname[entity_repr] = lasttok_to_repr[start-2]
                    except:
                        print("sentence index", sent_idx)
                        print(entities_in_sent)
                        print(lasttok_to_repr)
                        sys.exit(1)
                    print("Pruned abbreviation: %s" % (" ".join(tok_seq[start:end])))
                    entity_typ = ("ABBREV", "-")
                    abbrev = True
                entity_tok_set |= set(range(start, end))
                (new_start, new_end) = searchSeq(entity_repr.replace(" ", ""), tokenized, new_idx, aligned_new)
                if new_start == -1:
                    print("Something wrong with sentence %d" % sent_idx)
                    print(entity_repr)
                    print(tokenized)
                    sys.exit(1)
                map_repr = "%s##%s" % (entity_typ[0], entity_typ[1])
                # print new_start, new_end, map_repr
                lasttok_to_repr[end-1] = map_repr

                new_entity_spans.append("%d-%d####%s" % (new_start, new_end, map_repr))
                print("NER:", " ".join(tokenized[new_start:new_end]))
                # print aligned_repr
                curr_aligned = set(range(new_start, new_end))
                assert len(aligned_new & curr_aligned) == 0
                aligned_new |= curr_aligned
                new_idx = new_end

                aligned_set |= set(range(start, end))

            ##Align the first token, @someone
            if 0 not in aligned_set:
                first_tok = tok_seq[0]
                if len(first_tok) > 2 and first_tok[0] == '@':
                    entity_typ = ('NE_person', '-')
                    aligned_set.add(0)
                    (new_start, new_end) = searchSeq(first_tok, tokenized, new_idx)
                    if new_start == -1:
                        print("Something wrong with sentence %d" % sent_idx)
                        print(entity_repr)
                        print(tokenized)
                        sys.exit(1)
                    new_entity_spans.append("%d-%d####%s##%s" % (new_start, new_end, entity_typ[0], entity_typ[1]))
                    print("NER:", " ".join(tokenized[new_start:new_end]))

                    aligned_new |= set(range(new_start, new_end))

            new_idx = 0
            for start in range(n_toks):
                if start in aligned_set:
                    continue
                for end in range(start+7, start+1, -1):
                    if end > n_toks or end in aligned_set:
                        continue

                    entity_repr = " ".join(tok_seq[start:end])
                    if in_bracket(start, end, tok_seq) and (start-2) in entity_tok_set:
                        continue
                    if entity_repr in mle_map and mle_map[entity_repr][0] == "NONE":
                        continue
                    if entity_repr in entity_map or (entity_repr in mle_map and "NE_" in mle_map[entity_repr][0]):
                        if replaceSymbol(entity_repr) in mle_map:
                            new_name = replaceSymbol(entity_repr)
                            if new_name != entity_repr:
                                entity_repr = new_name
                                print("Replaced:", entity_repr)

                        if entity_repr in mle_map:
                            entity_typ = mle_map[entity_repr]
                            if entity_typ in aligned_repr:
                                print("Removed duplicate entity", entity_name)
                                entity_typ = ("REENT", "-")
                            aligned_repr.add(entity_typ)
                            if not "NE_" in entity_typ[0] and entity_repr.lower() in mle_map:
                                entity_typ = mle_map[entity_repr.lower()]
                                if "NE_" not in entity_typ[0]:
                                    print("Removed discovered entity:", entity_repr)
                                    continue
                            elif entity_typ[0] == "NONE":
                                print("Unaligned token:", entity_repr)
                                continue

                        elif start == 0 and entity_repr.lower() in mle_map:
                            entity_typ = mle_map[entity_repr.lower()]
                            assert entity_typ not in aligned_repr
                            if not "NE_" in entity_typ[0]:
                                print("Removed discovered entity:", entity_name)
                                continue
                        else:
                            assert entity_repr in entity_map
                            entity_name = entity_map[entity_repr]
                            if entity_name == "PER":
                                entity_typ = ('NE_person', '-')
                            else:
                                entity_name = ('NE', '-')

                        entity_repr = " ".join(tok_seq[start: end])
                        (new_start, new_end) = searchSeq(entity_repr.replace(" ", ""), tokenized, new_idx, aligned_new)
                        if new_start == -1:
                            print("Something wrong with sentence %d" % sent_idx)
                            print(entity_repr)
                            print(tokenized)
                            sys.exit(1)
                        new_entity_spans.append("%d-%d####%s##%s" % (new_start, new_end, entity_typ[0], entity_typ[1]))
                        # print "NER:", " ".join(tokenized[new_start:new_end])
                        curr_aligned = set(range(new_start, new_end))
                        assert len(aligned_new & curr_aligned) == 0
                        aligned_new |= curr_aligned
                        new_idx = new_end

                        aligned_set |= set(range(start, end))

            for idx in single_dates:
                new_idx = single_dates[idx]
                if idx not in aligned_set:
                    assert new_idx not in aligned_new
                    new_date_spans.append("%d-%d####%s" % (new_idx, new_idx+1, tokenized[new_idx]))
                    print("Single dates: %s" % tokenized[new_idx])
                else:
                    print("Filtered dates: %s, %s" % (tokenized[new_idx], tokenized))

            all_date_spans.append(new_date_spans)
            all_ner_spans.append(new_entity_spans)
        # for abbre_name in abbrev_to_fullname:
        #     print "%s ## %s" % (abbre_name, abbrev_to_fullname[abbre_name])
        saveSpanMap(all_date_spans, date_output)
        saveSpanMap(all_ner_spans, ner_output)
    else:
        # Segment sentence into spans: DATEs, NERs, phrases, NUMBERs, then single tokens.
        date_file = os.path.join(args.data_dir, "date.spans.txt")
        ner_file = os.path.join(args.data_dir, "ner.spans.txt")
        phrase_file = os.path.join(args.data_dir, "phrases")
        date_maps = utils.loadSpanMap(date_file, type="DATE")
        ner_maps = utils.loadSpanMap(ner_file)

        os.system("mkdir -p %s" % args.run_dir)

        output_tok = os.path.join(args.run_dir, "token")
        output_lem = os.path.join(args.run_dir, "lemma")
        output_pos = os.path.join(args.run_dir, "pos")
        output_ner = os.path.join(args.run_dir, "ner")
        json_output = os.path.join(args.run_dir, "decode.json")
        output_conll = os.path.join(args.run_dir, "concept")

        feat_dim = 97 # Currently fixed

        oracle_set = OracleData()

        tok_wf = open(output_tok, 'w')
        lemma_wf = open(output_lem, 'w')
        pos_wf = open(output_pos, 'w')
        ner_wf = open(output_ner, 'w')
        conll_wf = open(output_conll, 'w')

        # Map from a frequent phrase to its most frequent concept.
        phrase_map = utils.loadPhrases(phrase_file)

        for (sent_idx, tok_seq) in enumerate(toks):
            # if sent_idx != 579:
            #     continue
            lemma_seq, pos_seq, ner_seq = lems[sent_idx], poss[sent_idx], ners[sent_idx]
            ner_map, date_map = ner_maps[sent_idx], date_maps[sent_idx]

            concept_seq, category_seq, map_info_seq = [], [], []
            concept_to_word = []

            and_structure = check_and_sentences(" ".join(tok_seq))
            multi_tok_and = check_and_sentences(" ".join(tok_seq), tok_num=3)
            if and_structure:
                print("sentence index %d" % sent_idx)
                print(" ".join(tok_seq))

            # else:
            #     continue

            span_to_type = {}

            n_toks = len(tok_seq)

            start_to_end = {}
            for (start, end) in date_map:
                start_to_end[start] = end

            for (start, end) in ner_map:
                assert start not in start_to_end
                start_to_end[start] = end

            visited = set()
            collapsed_idx = -1
            if multi_tok_and:
                concept_seq.append("and")
                category_seq.append("and")
                if and_structure:
                    map_info_seq.append("AND||AND")
                else:
                    map_info_seq.append("AND1||AND1")
                concept_to_word.append(-1)
            for (tok_idx, curr_tok) in enumerate(tok_seq):
                if tok_idx in visited:
                    continue

                collapsed_idx += 1
                # First search for DATEs or NERs.
                if tok_idx in start_to_end:
                    end_idx = start_to_end[tok_idx]
                    if (tok_idx, end_idx) in date_map:
                        concept_seq.append("DATE")
                        category_seq.append("DATE")
                        map_info = "%s||NONE" % "_".join(tok_seq[tok_idx:end_idx])
                        map_info_seq.append(map_info)
                        span_to_type[(tok_idx, end_idx)] = (-1, -1, "DATE")
                        assert "_".join(tok_seq[tok_idx:end_idx]) == date_map[(tok_idx, end_idx)]
                    else:
                        ne_type, wiki_str = ner_map[(tok_idx, end_idx)]
                        if ne_type == "REENT" or ne_type == "ABBREV":
                            visited |= set(range(tok_idx, end_idx))
                            span_to_type[(tok_idx, end_idx)] = (-1, "-", "NE")
                            continue
                        concept_seq.append(ne_type)
                        category_seq.append(getCategories(ne_type, frequent_set))
                        if wiki_str == "-" or len(wiki_str.split("--")) != 2:
                            tok_repr = "_".join(tok_seq[tok_idx:end_idx]).replace("_@_-_@_", "-")
                        else:
                            assert len(wiki_str.split("--")) == 2, wiki_str
                            tok_repr = wiki_str.split("--")[0]
                            wiki_str = wiki_str.split("--")[1]
                        map_info = "%s||%s" % (tok_repr, wiki_str)
                        map_info_seq.append(map_info)
                        span_to_type[(tok_idx, end_idx)] = (-1, wiki_str, ne_type)
                    concept_to_word.append(collapsed_idx)

                    visited |= set(range(tok_idx, end_idx))
                    continue

                # Then we further check if there is any multi-token representations.
                for end_idx in range(n_toks, tok_idx, -1):
                    covered = set(range(tok_idx, end_idx))
                    if len(visited & covered) != 0:
                        continue

                    if identifyNumber(tok_seq[tok_idx:end_idx], mle_map):
                        concept_seq.append("NUMBER")
                        category_seq.append("NUMBER")
                        map_info = "%s||NONE" % "_".join(tok_seq[tok_idx:end_idx])
                        map_info_seq.append(map_info)
                        concept_to_word.append(collapsed_idx)
                        span_to_type[(tok_idx, end_idx)] = (-1, "NONE", "NUMBER")
                        visited |= set(range(tok_idx, end_idx))
                        break

                    tok_str = " ".join(tok_seq[tok_idx:end_idx])
                    lem_str = " ".join(lemma_seq[tok_idx:end_idx])
                    if tok_str in phrase_map or lem_str in phrase_map:
                        if tok_str in phrase_map:
                            concept_repr = phrase_map[tok_str]
                        else:
                            concept_repr = phrase_map[lem_str]
                        concept_seq.append(concept_repr)
                        category = getCategories(concept_repr, frequent_set)
                        category_seq.append(category)
                        map_info = "%s||%s" % ("_".join(tok_seq[tok_idx:end_idx]), concept_repr)
                        map_info_seq.append(map_info)
                        concept_to_word.append(collapsed_idx)
                        span_to_type[(tok_idx, end_idx)] = (-1, concept_repr, "PHRASE")
                        visited |= set(range(tok_idx, end_idx))
                        break

                if tok_idx in visited:
                    continue

                curr_tok = curr_tok.lower()
                curr_lem, curr_pos, curr_ner = lemma_seq[tok_idx].lower(), pos_seq[tok_idx], ner_seq[tok_idx]
                if curr_tok in mle_map or curr_lem in mleLemmaMap:
                    if curr_tok in mle_map:
                        category, concept_repr = mle_map[curr_tok]
                    else:
                        category, concept_repr = mleLemmaMap[curr_lem]
                    if category != "NONE" and not utils.entity_category(category):
                        span_to_type[(tok_idx, tok_idx+1)] = (-1, concept_repr, category)

                        if utils.special_categories(category):
                            concept_seq.append(category)
                        else:
                            concept_seq.append(concept_repr)
                        map_repr = concept_repr
                        if concept_repr[:3] == "NEG":
                            map_repr = "%s polarity:-" % concept_repr[4:]
                        map_info = "%s||%s" % (curr_tok, map_repr)
                        category = getCategories(category, frequent_set)
                        category_seq.append(category)
                        map_info_seq.append(map_info)
                        concept_to_word.append(collapsed_idx)
                    visited.add(tok_idx)
                elif curr_tok in VERB_LIST or curr_lem in VERB_LIST:
                    if curr_tok in VERB_LIST:
                        subgraph = VERB_LIST[curr_tok][0]
                    else:
                        subgraph = VERB_LIST[curr_lem][0]
                    subgraph_repr = alignment_utils.subgraph_str(subgraph)
                    print("Retrieved: %s -> %s" % (curr_tok, subgraph_repr))
                    root_repr = subgraph_repr.split()[0].strip()
                    category = "MULT_%s" % root_repr
                    span_to_type[(tok_idx, tok_idx+1)] = (-1, subgraph_repr, category)
                    concept_seq.append(category)
                    category = getCategories(category, frequent_set)
                    category_seq.append(category)
                    map_info = "%s||%s" % (curr_tok, subgraph_repr)
                    map_info_seq.append(map_info)
                    concept_to_word.append(collapsed_idx)
                elif not utils.allSymbols(curr_tok):
                    curr_repr = curr_lem
                    if curr_pos[0] == "V":
                        curr_repr = "%s-01" % curr_lem
                    curr_category = curr_repr
                    span_to_type[(tok_idx, tok_idx+1)] = (-1, curr_repr, curr_category)
                    concept_seq.append(curr_repr)
                    category = getCategories(curr_repr, frequent_set)
                    category_seq.append(category)
                    map_info = "%s||%s" % (curr_tok, curr_repr)
                    map_info_seq.append(map_info)
                    concept_to_word.append(collapsed_idx)

            # print span_to_type
            collapsed_toks, collapsed_lem, collapsed_pos, collapsed_ner, _, _ = collapseTokens(
                tok_seq, lemma_seq, pos_seq, ner_seq, span_to_type, False, 20)
            assert len(collapsed_toks) == collapsed_idx + 1, "%s:%d:%d" % (str(collapsed_toks), sent_idx,
                                                                           collapsed_idx+1)
            assert len(concept_seq) == len(category_seq)
            assert len(concept_seq) == len(map_info_seq)
            assert len(concept_seq) == len(concept_to_word)

            print(" ".join(collapsed_toks), file=tok_wf)
            print(" ".join(collapsed_lem), file=lemma_wf)
            print(" ".join(collapsed_pos), file=pos_wf)
            print(" ".join(collapsed_ner), file=ner_wf)
            print('sentence %d' % sent_idx, file=conll_wf)
            for (concept_idx, concept) in enumerate(concept_seq):
                align_widx = concept_to_word[concept_idx]
                print("%d\t%s\t%s\t%s\t%s" % (
                    concept_idx, concept, category_seq[concept_idx], map_info_seq[concept_idx],
                    collapsed_toks[align_widx]), file=conll_wf)
            print("", file=conll_wf)

            # Next we prepare the input files for the decoder.
            oracle_seq = ["SHIFT", "POP"] * len(concept_seq)
            num_actions = len(oracle_seq) # "SHIFT", "POP" for each concept
            feat_seq = [[""] * feat_dim for _ in range(num_actions)]
            focus_align = [(-1, -1)] * num_actions
            oracle_example = OracleExample(collapsed_toks, collapsed_lem, collapsed_pos, collapsed_ner,
                                           concept_seq, category_seq, map_info_seq, feat_seq,
                                           oracle_seq, focus_align, concept_to_word)
            oracle_set.addExample(oracle_example)

        tok_wf.close()
        lemma_wf.close()
        pos_wf.close()
        ner_wf.close()
        conll_wf.close()
        oracle_set.toJSON(json_output)

        # all_spans = sorted(all_spans, key=lambda span: (span[0], span[1]))

        # collapsed_toks, collapsed_lem, collapsed_pos, _ = collapseTokens(tok_seq, lem_seq, pos_seq, span_to_type, False)

def buildLinearEnt(entity_name, ops):
    ops_strs = ['op%d( %s )op%d' % (index, s, index) for (index, s) in enumerate(ops, 1)]
    ent_repr = '%s name( name %s )name' % (entity_name, ' '.join(ops_strs))
    return ent_repr


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type=str, help="the task to run", required=True)

    argparser.add_argument("--amr_file", type=str, help="the original AMR graph files", required=False)
    argparser.add_argument("--input_file", type=str, help="the original sentence file", required=False)
    argparser.add_argument("--token_file", type=str, help="the tokenized sentence file", required=False)
    argparser.add_argument("--align_file", type=str, help="the original AMR graph files", required=False)
    argparser.add_argument("--output", type=str, help="the output file", required=False)
    argparser.add_argument("--conll_file", type=str, help="output the AMR graph in conll format", required=False)
    argparser.add_argument("--dev_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--test_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--dev_output", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--test_output", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--stop", type=str, help="stop words file", required=False)
    argparser.add_argument("--lemma", type=str, help="lemma file", required=False)
    argparser.add_argument("--data_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--map_file", type=str, help="map file from training")
    argparser.add_argument("--date_file", type=str, help="all the date in each sentence")
    argparser.add_argument("--dep_file", type=str, help="dependency file")
    argparser.add_argument("--run_dir", type=str, help="the output directory for saving the constructed forest")
    argparser.add_argument("--table_dir", type=str, help="the frequency table for various statistics")
    argparser.add_argument("--use_lemma", action="store_true", help="if use lemmatized tokens")
    argparser.add_argument("--parallel", action="store_true", help="if to linearize parallel sequences")
    argparser.add_argument("--realign", action="store_true", help="if to realign the data")
    argparser.add_argument("--use_stats", action="store_true", help="if use a built-up statistics")
    argparser.add_argument("--save_mode", action="store_true", help="if to update entity stats")
    argparser.add_argument("--stats_dir", type=str, help="the statistics directory")
    argparser.add_argument("--freq_dir", type=str, help="the training frequency statistics")
    argparser.add_argument("--min_prd_freq", type=int, default=50, help="threshold for filtering predicates")
    argparser.add_argument("--min_var_freq", type=int, default=50, help="threshold for filtering non predicate variables")
    argparser.add_argument("--index_unknown", action="store_true", help="if to index the unknown predicates or non predicate variables")

    args = argparser.parse_args()
    # dependency_align(args.dep_file, args.token_file)
    if args.task == "realign":
        realign(args.input_file, args.token_file, args.align_file, args.output)
    elif args.task == "reformat":
        linearize_amr(args)
    else:
        linearizeData(args, args.data_dir, args.freq_dir, args.output, args.save_mode)

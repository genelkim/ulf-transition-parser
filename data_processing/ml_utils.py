from collections import defaultdict
import csv
import pickle
from numpy import zeros, ones, array, tile
import sys, os, re

def read_toks(filename):
    tok_seqs = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                toks = line.split()
                tok_seqs.append(toks)
    return tok_seqs

def load_mle(path, tok_to_concept):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                splits = line.strip().split("####")
                tok_str = splits[0].strip()
                concept_str = splits[1].split("##")[0]
                if tok_str not in tok_to_concept:
                    tok_to_concept[tok_str] = concept_str
        f.close()

def loadConceptIDCounts(path, threshold=0.30, min_count=40, delimiter="\t"):
    def has_alphabet(s):
        match = re.search("[a-zA-Z0-9]", s)
        if match:
            return True
        return False

    ambiguous_set = set()

    for line in open(path):
        if line.strip():
            fields = line.strip().split(delimiter)
            word = fields[0].strip()

            if word == "NE" or word[:3] == "NE_":
                continue

            choices = set()

            total_count = int(fields[1].strip())
            if total_count < min_count:
                continue
            total_count *= (1-threshold)

            curr_count = 0

            for s in fields[2].split(";"):
                concept = ":".join(s.split(':')[:-1])
                count = int(s.split(':')[-1])

                category = concept.split("||")[0]
                target = concept.split("||")[1].strip()
                if concept[:3] == "NE_" or category == "DATE" or category == "NUMBER":
                    target = category

                choices.add(target)

                curr_count += count
                if curr_count > total_count:
                    break
            if len(choices) > 1:
                ambiguous_set.add(word)
    return ambiguous_set

def loadCandidateMap(conceptIDFile, lemConceptIDFile, labelIndexer, delimiter="\t", threshold=0.10):
    def has_alphabet(s):
        match = re.search("[a-zA-Z0-9]", s)
        if match:
            return True
        return False
    word2choices = defaultdict(set)
    lemma2choices = defaultdict(set)
    for line in open(conceptIDFile):
        if line.strip():
            fields = line.strip().split(delimiter)
            word = fields[0].strip()

            total_count = int(fields[1].strip())
            ignore = total_count < 5 and has_alphabet(word)
            total_count *= (1-threshold)

            curr_count = 0

            for s in fields[2].split(";"):
                concept = ":".join(s.split(':')[:-1])
                count = int(s.split(':')[-1])

                # target = "conID:" + concept
                target = concept.split("||")[1].strip()
                if concept[:3] == "NE_":
                    target = concept.split("||")[0]

                if ignore and target == "NONE":
                    print("Disallow mapping %s to NONE" % word)

                if target not in labelIndexer:
                    print("not in label indexer:", word, target)
                    continue

                l = labelIndexer[target]
                word2choices[word].add(l)
                curr_count += count
                if curr_count > total_count:
                    break

    for line in open(lemConceptIDFile):
        if line.strip():
            fields = line.strip().split(delimiter)
            lemma = fields[0].strip()

            total_count = int(fields[1].strip())
            ignore = total_count < 5 and has_alphabet(lemma)
            total_count *= (1-threshold)
            curr_count = 0

            for s in fields[2].split(";"):
                concept = ":".join(s.split(':')[:-1])
                count = int(s.split(':')[-1])

                target = concept.split("||")[1].strip()
                if concept[:3] == "NE_":
                    target = concept.split("||")[0]
                if ignore and target == "NONE":
                    print("Disallow mapping %s to NONE" % lemma)

                if target not in labelIndexer:
                    print("not in label indexer:", word, target)
                    continue

                l = labelIndexer[target]
                lemma2choices[lemma].add(l)
                curr_count += count
                if curr_count > total_count:
                    break

    return word2choices, lemma2choices

def load_data(filename, featIndexer, labelIndexer, emissionWeights, output_choices, is_train = True, delimiter="\t"):
    X = []
    X_t = []
    Y = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            fields = line.split(delimiter)
            curr_y = labelIndexer.setdefault(fields[0], len(labelIndexer))
            if is_train:
                output_choices.add(fields[0])
            Y.append(curr_y)

            x_s = []
            curr_word = None
            curr_lem = None
            curr_cate = None

            for x_feat in fields[1:]:
                if x_feat.startswith('tok='):
                    curr_word = x_feat.replace('tok=', '')
                if x_feat.startswith('lem='):
                    curr_lem = x_feat.replace('lem=', '')
                if x_feat.startswith('tok0='):
                    curr_cate = x_feat.replace('tok0=', '')

                if x_feat in featIndexer:
                    curr_x = featIndexer[x_feat]
                else:
                    curr_x = featIndexer.setdefault(x_feat, len(featIndexer))
                    assert len(emissionWeights) == curr_x
                    emissionWeights.append(defaultdict(int))

                x_s.append(curr_x)

            X.append(x_s)
            assert curr_word and curr_lem and curr_cate, '%s %s %s' % (curr_word, curr_lem, curr_cate)
            X_t.append((curr_word, curr_lem, curr_cate))

    return (X, X_t, Y, emissionWeights)


def viterbi_emission(x_s, emissionWeights, cand_set):
    ret = 0

    max_score = float("-inf")
    for curr_y in cand_set:
        curr_score = sum([emissionWeights[curr_x][curr_y] for curr_x in x_s])
        if curr_score > max_score:
            max_score = curr_score
            ret = curr_y
    return ret

#Make sure insert a singleton start label set
def viterbi_transition(x_seq, emissionWeights, transitionWeights, scores, backpointers, y_cands, SOS):
    N = len(x_seq)
    path = [None for x in range(N)]

    scores[0][SOS] = 0

    for i in range(1, N+1): #Each position is just a maximization over emission scores
        x_s = x_seq[i-1]   #A list of feature indices
        cand_set = y_cands[i]
        for curr_y in cand_set:
            emission_score = sum([emissionWeights[curr_x][curr_y] for curr_x in x_s])
            prev_cand = y_cands[i-1]
            max_score = float("-inf")
            for prev_y in prev_cand:
                curr_score = scores[i-1][prev_y] + transitionWeights[curr_y][prev_y]
                if curr_score > max_score:
                    max_score = curr_score
                    backpointers[i][curr_y] = prev_y
            scores[i][curr_y] = max_score + emission_score

    max_score = float("-inf")
    #print scores
    #print y_cands[N]
    for last_y in y_cands[N]:
        if scores[N][last_y] > max_score:
            max_score = scores[N][last_y]
            path[-1] = last_y

    for i in range(N-1, 0, -1):
        next_y = path[i]
        path[i-1] = backpointers[i+1][next_y]

    #print path
    return path

def check_loss(y_hat, y_gold):
    for (h, y) in zip(y_hat, y_gold):
        if h != y:
            return True
    return False

def viterbi(x, emissionWeights, transitionWeights, firstWeights, tags):
    """Use the viterbi algorithm to find best tag sequence for a sentence given an HMM model.

    """

    N = len(x)
    num_tags = len(tags)

    scores = [None for i in range(N)]
    backpointer = [None for i in range(N)]

    #Deal with the first row separately
    try:
        scores[0] = firstWeights + emissionWeights[x[0]]
    except:
        print(scores)
        print(firstWeights)
        print(emissionWeights[x[0]])
        sys.exit(-1)

    # For each observation (word) in sequence
    for i in range(1, N):
        all_scores = tile(scores[i-1].reshape(num_tags,1), (1, num_tags)) + transitionWeights
        scores[i] = emissionWeights[x[i]] + all_scores.max(axis=0)
        backpointer[i] = all_scores.argmax(axis=0)

    last_tag = scores[N-1].argmax()
    path = [last_tag]
    curr_p = last_tag
    for i in range(N-1, 0, -1):
        path.append(backpointer[i][curr_p])
        curr_p = backpointer[i][curr_p]
    path.reverse()

    return path


def test_accuracy(X, Y, emissionWeights, transitionWeights, firstWeights, tags):
    """
    Test accuracy of viterbi algorithm on given data using given emission & transmission weights.

    """
    score = 0.0
    total_tags = float(sum(len(y) for y in Y))
    correct_tags = 0.0
    for x, y in zip(X, Y):

        y_hat = viterbi(x, emissionWeights, transitionWeights, firstWeights, tags)
        correct_tags += (y_hat == array(y)).sum()

    return correct_tags / total_tags

def save_score(fname, i, score):
    with open(fname, 'a') as w:
        writer = csv.writer(w)
        writer.writerow([i, score])


from scipy.special import logsumexp
# from utils import viterbi, test_accuracy, save_score, load_data, load_mappings, read_toks
from ml_utils import *
import argparse
import time
import sys
import os
from constants import *
import re
import random

def run_perceptron(args):
    def loadMap(file, threshold):
        curr_f = open(file, 'r')
        concept2choices = {}
        for line in curr_f:
            if line.strip():
                fields = line.strip().split()
                assert len(fields) == 2
                totalcount = 0
                choices = set()
                for r in fields[1].split(";"):
                    l = r.split(":")[0]
                    count = int(r.split(":")[1])
                    totalcount += count

                ceiling = totalcount * (1-threshold)
                count = 0
                for r in fields[1].split(";"):
                    if count > ceiling:
                        break
                    count += int(r.split(":")[1])

                    l = r.split(":")[0]
                    choices.add(l)
                concept2choices[fields[0]] = choices
        return concept2choices

    #Construct the candidate set for each classification
    def buildChoices(X, X_t, labelIndexer, table_dir):
        conceptID_file = os.path.join(table_dir, "conceptIDcounts.txt")
        lemmaID_file = os.path.join(table_dir, "lemmaIDcounts.txt")
        word2choices, lemma2choices = loadCandidateMap(conceptID_file, lemmaID_file, labelIndexer)

        labelchoices = []

        n_instances = len(X)
        for sent_index in range(n_instances):
            (word, lem, category) = X_t[sent_index]
            if category == "NE":
                y_cands = word2choices[category]
            elif word in word2choices:
                y_cands = word2choices[word]
            elif lem in lemma2choices:
                y_cands = lemma2choices[lem]
            elif category == "DATE" or category == "NUMBER":
                y_cands = set([labelIndexer[category]])
            else:
                print ("unseen: %s %s %s" % (word, lem, category))
                y_cands = set([labelIndexer['-NULL-']])   #Just pretend to predict the first label always
            labelchoices.append(y_cands)

        return labelchoices

    def saveIndexer(indexer, file):
        with open(file, 'w') as wf:
            print(len(indexer), file=wf)
            for l in indexer:
                print(("%s %d" % (l, indexer[l])), file=wf)
            wf.close()

    def saveEmissionWeights(emissionWeights, featIndexer, labelIndexer, file):
        assert len(emissionWeights) == len(featIndexer)
        observe_length = len(featIndexer)
        target_length = len(labelIndexer)

        with open(file, 'w') as wf:
            print(observe_length, target_length, file=wf)
            for x_id in range(observe_length):
                for y_id in emissionWeights[x_id]:
                    if emissionWeights[x_id][y_id] != 0:
                        print(x_id, y_id, emissionWeights[x_id][y_id], file=wf)
            wf.close()

    featIndexer = {}
    labelIndexer = {}

    emissionWeights = [] #Emission weights is a table of maps for each feature
    averageEmissionWeights = []

    output_choices = set()
    (X_train, X_t_train, Y_train, emissionWeights) = load_data(args.train, featIndexer, labelIndexer, emissionWeights,
                                                               output_choices, True)
    (X_dev, X_t_dev, Y_dev, emissionWeights) = load_data(args.dev, featIndexer, labelIndexer, emissionWeights,
                                                         output_choices, False)

    # First dimension is the feature index.
    n_emission = len(emissionWeights)
    for i in range(n_emission):
        averageEmissionWeights.append(defaultdict(int))

    print('output choices', output_choices)
    choices_wf = open(("%s_output_choices.txt" % args.task), 'w')
    for l in output_choices:
        print(l, file=choices_wf)
    choices_wf.close()
    if "-NULL-" not in labelIndexer:
        labelIndexer["-NULL-"] = len(labelIndexer)
    print(len(labelIndexer))
    print(labelIndexer)

    featureindex_file = os.path.join(args.result_dir, 'feat_index.txt')
    labelindex_file = os.path.join(args.result_dir, 'label_index.txt')

    saveIndexer(featIndexer, featureindex_file)
    saveIndexer(labelIndexer, labelindex_file)

    train_choices = buildChoices(X_train, X_t_train, labelIndexer, "./train_tables")
    dev_choices = buildChoices(X_dev, X_t_dev, labelIndexer, "./train_tables")

    assert len(train_choices) == len(X_train)
    assert len(dev_choices) == len(X_dev)

    if args.test:
        (X_test, X_t_test, _, emissionWeights) = load_data(args.test, featIndexer, labelIndexer, emissionWeights)

    assert len(emissionWeights) == len(featIndexer)
    print('finished loading the data')

    print('A total of %d features' % len(featIndexer))
    print('A total of %d labels' % len(labelIndexer))

    print('%d train sentences' % len(X_train))
    print('%d dev sentences' % len(X_dev))

    prev_acc = 0.0
    emissionweight_file = os.path.join(args.result_dir, 'emissions.txt')

    num_feats = len(featIndexer)
    num_labels = len(labelIndexer)

    labelList = [None for x in range(num_labels+1)]

    target_set = set()
    for l in labelIndexer:
        target_set.add(labelIndexer[l])

    for s in labelIndexer:
        value = labelIndexer[s]
        labelList[value] = s

    if args.use_transition:
        print('Use transition')
        transitionWeights = []
        for i in range(num_labels):
            transitionWeights.append(defaultdict(int))

        assert 'SOS' not in labelIndexer
        SOS = labelIndexer.setdefault('SOS', len(labelIndexer))
        scores = []
        backpointers = []
        for i in range(300):
            scores.append(defaultdict(int))
            backpointers.append(defaultdict(int))

    print('labelIndexer length: %d' % len(labelIndexer))

    random.seed(20170412)
    for iter in range(args.iters):
        print('iteration %d'% iter)
        seq = range(len(X_train))
        random.shuffle(seq)
        sys.stdout.flush()

        start_time = time.time()
        for curr_index in range(len(X_train)):
            sent_no = seq[curr_index]
            x_seq = X_train[sent_no]

            y = Y_train[sent_no]

            if train_choices:
                y_cands = train_choices[sent_no]
            else:
                y_cands = target_set

            if args.use_transition:
                y_hat = viterbi_transition(x_seq, emissionWeights, transitionWeights, scores, backpointers, y_cands, SOS)
            else:
                y_hat = viterbi_emission(x_seq, emissionWeights, y_cands)

            if y_hat != y:
                for x in x_seq:
                    emissionWeights[x][y] += 1
                    emissionWeights[x][y_hat] -= 1

        #Evaluate on dev
        accurate_toks = 0
        total_toks = 0

        for (sent_no, (x_seq, x_t_seq, y)) in enumerate(zip(X_dev, X_t_dev, Y_dev)):
            if dev_choices:
                y_cands = dev_choices[sent_no]
            else:
                y_cands = target_set

            if args.use_transition:
                y_hat = viterbi_transition(x_seq, emissionWeights, transitionWeights, scores, backpointers, y_cands, SOS)
            else:
                y_hat = viterbi_emission(x_seq, emissionWeights, y_cands)

            total_toks += 1

            if y_hat == y:
                accurate_toks += 1

        curr_acc = float(accurate_toks) / float(total_toks)
        print('Accuracy on dev for the current iteration is %f' % (float(accurate_toks)/ float(total_toks)))

        if curr_acc > prev_acc:
            saveEmissionWeights(emissionWeights, featIndexer, labelIndexer, emissionweight_file)
            prev_acc = curr_acc

        #Evaluate with averaging
        for i in range(n_emission):
            for curr_y in emissionWeights[i]:
                averageEmissionWeights[i][curr_y] += emissionWeights[i][curr_y]

        accurate_toks = 0
        total_toks = 0

        for (sent_no, (x_seq, x_t_seq, y)) in enumerate(zip(X_dev, X_t_dev, Y_dev)):
            if dev_choices:
                y_cands = dev_choices[sent_no]

            else:
                y_cands = target_set

            y_hat = viterbi_emission(x_seq, averageEmissionWeights, y_cands)

            total_toks += 1
            if y_hat == y:
                accurate_toks += 1


        curr_acc = float(accurate_toks) / float(total_toks)
        print('Accuracy on dev for the current iteration is %f' % (float(accurate_toks)/ float(total_toks)))

        if curr_acc > prev_acc:
            saveEmissionWeights(averageEmissionWeights, featIndexer, labelIndexer, emissionweight_file)
            prev_acc = curr_acc

        #Evaluate on test
        if args.test:
            result_file = os.path.join(args.result_dir, 'result_%d' % iter)
            result_f = open(result_file, 'w')
            #for (x_seq, x_t_seq, z_seq) in zip(X_test, X_t_test, Z_test):
            #    assert len(x_seq) == len(z_seq), '%d %d %d' % (len(x_seq), len(z_seq))
            #    y_cands = []
            #    if args.use_transition:
            #        y_cands.append(start_set)

                # for (index, (span, is_pred, is_ent)) in enumerate(z_seq):
                #     (word, lem) = x_t_seq[index]
                #     curr_cand_set = generate_cand_set(non_map_words, stop_words, entity_label_set, UNKNOWN, is_ent, is_pred, word, lem, arg_label_set, word_label_set, lem_label_set, none_set, unknown_set)
                #     y_cands.append(curr_cand_set)
                #     #curr_cand_set = set()

                #     #(tok, lem) = x_t_seq[index]
                #     #if tok in unknown_set:
                #     #    curr_cand_set.add(UNKNOWN)

                #     #if is_pred:
                #     #    curr_cand_set = default_label_set

                #     #if tok in word_label_set:
                #     #    y_cands.append(word_label_set[tok] | curr_cand_set)

                #     #elif lem in lem_label_set:
                #     #    y_cands.append(lem_label_set[lem] | curr_cand_set)
                #     #else:
                #     #    y_cands.append(none_set | curr_cand_set)

                # if args.use_transition:
                #     y_hat_seq = viterbi_transition(x_seq, emissionWeights, transitionWeights, scores, backpointers, y_cands, SOS)
                # else:
                #     y_hat_seq = viterbi_emission(x_seq, emissionWeights, y_cands)

                # spans = [x for (x, y, z) in z_seq]
                # for (label_index, span) in zip(y_hat_seq, spans):
                #     print >>result_f, '%s %s' % (labelList[label_index], span)
                # print >>result_f, ''

            result_f.close()
        print('time for current iteration: ', time.time() - start_time)
        sys.stdout.flush()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--iters", type=int, default=10, help="Number of iterations to run learning algorithm with", required=False)
    parser.add_argument("-l", "--learning_rate", type=float, help="learning rate", default=0.1, required=False)
    parser.add_argument("-s", "--sample_rate", type=float, help="learning rate", default=0.5, required=False)
    parser.add_argument("--train", type=str, help="training data", required=True)
    parser.add_argument("--dev", type=str, help="dev data", required=True)
    parser.add_argument("--test", type=str, help="test data", required=False)
    parser.add_argument("--result_dir", type=str, help="result directory", required=False)
    parser.add_argument("--task", type=str, help="current task", required=False)
    parser.add_argument("--word", type=str, help="dev word file", required=False)
    parser.add_argument("--lemma", type=str, help="dev lemma file", required=False)
    parser.add_argument("--pos", type=str, help="dev POS file", required=False)
    parser.add_argument("--use_transition", action="store_true", help="if to use transition scores to compute the best path")
    parser.add_argument("--candidate", action="store_true", help="if to use transition scores to compute the best path")

    args = parser.parse_args()

    run_perceptron(args)

#!/usr/bin/python
import sys
import os
from lemma_util import initialize_lemma
import argparse

#Use the lemma dict to find the lemma of the word
def get_lemma(word, pos, infl_lemma, trans_lemma, der_lemma=None):
    if pos[0] in 'NV' and word in infl_lemma: #Noun or verb
        return list(infl_lemma[word])[0]
    if pos[0] == 'V' and word in trans_lemma: #Verb
        return list(trans_lemma[word])[0]
    if pos == "RB" and der_lemma is not None and word in der_lemma:
        return list(der_lemma[word])[0]
    return word

def main(args):
    infl_file = os.path.join(args.lemma_dir, 'infl.lemma')
    trans_file = os.path.join(args.lemma_dir, 'trans.lemma')
    der_file = os.path.join(args.lemma_dir, 'der.lemma')

    infl_lemma = initialize_lemma(infl_file)
    trans_lemma = initialize_lemma(trans_file)
    der_lemma = initialize_lemma(der_file)

    tok_file = None
    pos_file = None
    for curr_file in os.listdir(args.data_dir):
        abs_file = os.path.join(args.data_dir, curr_file)
        if not os.path.isdir(abs_file):
            if "token" in curr_file:  # Naming constraint
                tok_file = abs_file
            if "pos" in curr_file:
                pos_file = abs_file
    assert tok_file is not None, "No token file assigned"
    assert pos_file is not None, "No POS file assigned"

    tok_seqs = [line.strip().split() for line in open(tok_file, 'r')]
    pos_seqs = [line.strip().split() for line in open(pos_file, 'r')]

    result_file = os.path.join(args.data_dir, 'lemma')
    with open(result_file, 'w') as wf:
        for (toks, poss) in zip(tok_seqs, pos_seqs):
            assert len(toks) == len(poss)

            lemmas = []
            for (word, pos) in zip(toks, poss):
                lem = get_lemma(word.lower(), pos, infl_lemma, trans_lemma, der_lemma)
                lemmas.append(lem)

            print(' '.join(lemmas), file=wf)

        wf.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--lemma_dir", type=str, help="lemma files directory", required=False)
    argparser.add_argument("--num_files", type=int, help="number of files")

    args = argparser.parse_args()
    main(args)

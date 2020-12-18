from collections import defaultdict
def initialize_lemma(lemma_file):
    lemma_map = defaultdict(set)
    with open(lemma_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            word = fields[0]
            lemma = fields[1]
            if word == lemma:
                continue
            lemma_map[word].add(lemma)
    return lemma_map


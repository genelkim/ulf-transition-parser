#!/usr/bin/python3
import re, sys, os
import pickle
from identify_entity import entities_inline

def identify_entities(tok_file, ner_file, mle_map):
    def all_nonent(toks):
        for tok in toks:
            if tok in mle_map:
                entity_typ = mle_map[tok]
                if "NE_" in entity_typ[0]:
                    return False
            elif tok.lower() in mle_map:
                entity_typ = mle_map[tok.lower()]
                if "NE_" in entity_typ[0]:
                    return False
            else:
                return False
        return True

    all_entities = []
    stop_words = set(["of", "is", "in", "for"])

    entity_map = {}  #For recognizing entities in the same document.
    with open(tok_file, 'r') as tok_f:
        with open(ner_file, 'r') as ner_f:
            for (i, tok_line) in enumerate(tok_f):
                sent_entities = []
                ner_line = ner_f.readline()
                tok_line = tok_line.strip()
                if tok_line:
                    aligned_toks = set()

                    toks = tok_line.split()
                    matched_entities = entities_inline(ner_line)
                    entities = {}
                    for (role, ent) in matched_entities:
                        ent = ent.replace('_', ' ')
                        #ent = ent.lower()

                        #if len(ent.split()) == 1 and all_nonent(ent.split()):
                        #    print "Filtered entity:", ent
                        #    continue
                        entities[ent] = role
                        if ent not in mle_map and (not ent.split()[-1] in stop_words) and (not ent.split()[0] in stop_words):
                            entity_map[ent] = role

                    length = len(toks)
                    for start in range(length):
                        if start in aligned_toks:
                            continue

                        for span in range(length+1, 0, -1):
                            end = start + span
                            if end-1 in aligned_toks:
                                continue

                            if end > length:
                                continue

                            #curr_str = (' '.join(toks[start:end])).lower()
                            curr_str = ' '.join(toks[start:end])

                            if curr_str in entities:
                                if toks[start] in stop_words or toks[end-1] in stop_words:
                                    print("entities with stopword:", curr_str)
                                curr_set = set(range(start, end))
                                aligned_toks |= curr_set
                                # print curr_str
                                sent_entities.append((start, end, entities[curr_str]))

                            elif curr_str in mle_map:
                                cur_symbol, wiki_label = mle_map[curr_str]
                                if (not "NE_" in cur_symbol) or cur_symbol == "NONE":
                                    continue
                                curr_set = set(range(start, end))
                                aligned_toks |= curr_set
                                # print curr_str
                                sent_entities.append((start, end, cur_symbol))
                all_entities.append(sent_entities)
    return all_entities, entity_map

def load_entities(stats_dir):
    stats_files = os.popen('ls %s/stat_*' % stats_dir).read().split()
    all_entities = set()

    all_non_pred = set()
    all_pred = set()

    assert len(stats_files) == 8
    for file in stats_files:
        f = open(file, 'rb')
        stats = pickle.load(f)
        entity_words = stats[3]

        all_pred |= stats[0]
        all_non_pred |= set(stats[6].keys())

        all_entities |= entity_words
        f.close()

    all_entities -= all_non_pred
    all_entities -= all_pred
    #for entity in all_entities:
    #    if entity in all_non_pred:
    #        continue
    #    if entity in all_pred:
    #        continue
    #    #print entity

    return all_entities

if __name__ == '__main__':
    all_entities = load_entities(sys.argv[3])
    identify_entities(sys.argv[1], sys.argv[2], all_entities)

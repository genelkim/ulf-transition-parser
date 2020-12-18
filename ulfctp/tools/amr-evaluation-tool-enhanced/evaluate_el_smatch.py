"""
EL-smatch score of each pair of EL-AMR formulas and document score.

Inputs:
  [hypothesis EL-AMR formula file]
  [gold EL-AMR formula file]
  [output file]

Outputs:
  Results stored in output file.

Input format:
  Each EL-AMR formula is separated by a blank line.

Output format:
  Tab separated [precision, recall, F-score] for each pair.
  After "===============" the document precision, recall, and
  F-score are printed.
"""

from __future__ import division
# TODO: change absl flags to argparse (that's available in BlueHive alreayd)
import argparse
import sys
import el_smatch.el_amr as el_amr
import el_smatch.el_smatch as el_smatch
from collections import namedtuple

# Error log location
ERROR_LOG = sys.stderr

# Debug log location
DEBUG_LOG = sys.stderr


VERBOSE = False
ELSmatchMetrics = namedtuple('ELSmatchMetrics', ['precision', 'recall', 'f1', 'matched_triples', 'test_triples', 'gold_triples'])

"""
Runs EL-Smatch on the amr given amr info for n iterations.
Records the best-mapping for each run, matching triple number for the mapping
and the number of iterations to converge for each run.
"""
def el_smatch_run(instance1, attribute1, relation1, 
                  instance2, attribute2, relation2,
                  prefix1, prefix2, n):
  if VERBOSE:
    print("Running EL-Smatch")
  # Compute candidate pool - all possible node match candidates.
  # In the hill-climbing, we only consider candidate in this pool to save computing time.
  # weight_dict is a dictionary that maps a pair of node
  (candidate_mappings, weight_dict) = \
    el_smatch.compute_pool(instance1, attribute1, relation1,
                           instance2, attribute2, relation2,
                           prefix1, prefix2)
 
  # Run iterations.
  best_mapping = None
  best_match_num = -1

  for i in range(0, n):
    if VERBOSE:
      print("iteration {}".format(i))
    if i == 0:
      # smart initialization used for the first round
      cur_mapping = el_smatch.smart_init_mapping(candidate_mappings, instance1, instance2)
    else:
      # random initialization for the other round
      cur_mapping = el_smatch.random_init_mapping(candidate_mappings)
    # compute current triple match number
    match_num = el_smatch.compute_match(cur_mapping, weight_dict)
    
    steps = 0
    while True:
      steps += 1
      # get best gain
      (gain, new_mapping) = el_smatch.get_best_gain(cur_mapping, candidate_mappings, weight_dict,
                        len(instance2), match_num)
      # hill-climbing until there will be no gain for new node mapping
      if gain <= 0:
        break
      # otherwise update match_num and mapping
      match_num += gain
      cur_mapping = new_mapping[:]
    
    # Record mapping and steps.
    if match_num > best_match_num:
      best_match_num = match_num
      best_mapping = cur_mapping[:]

  return (best_mapping, best_match_num)


"""
Scores an EL-Smatch mapping.
Returns a 6-tuple:
(precision, recall, f_score, match_triple_num, test_triple_num, gold_triple_num)
"""
def score_el_smatch_mapping(mapping, match_num, 
                            instance1, attribute1, relation1, 
                            instance2, attribute2, relation2):
  
  test_triple_num = len(instance1) + len(attribute1) + len(relation1)
  gold_triple_num = len(instance2) + len(attribute2) + len(relation2)
  
  # if each AMR pair should have a score, compute and output it here
  (precision, recall, f_score) = el_smatch.compute_f(match_num,
                                                     test_triple_num,
                                                     gold_triple_num)
  return ELSmatchMetrics(precision, recall, f_score, match_num, test_triple_num, gold_triple_num)


"""
Calculates the the document precision, recall, f_score, match_num, test_num,
and gold_num given the list of scores for each pair in the same format.
"""
def calculate_document_score(pair_scores):
  # p,r,f,m,t,g : precision, recall, f-score, match triples, test triples, gold triples.
  match_num = sum([m for p,r,f,m,t,g in pair_scores])
  test_num = sum([t for p,r,f,m,t,g in pair_scores])
  gold_num = sum([g for p,r,f,m,t,g in pair_scores])

  precision, recall, f_score = \
      el_smatch.compute_f(match_num, test_num, gold_num)
  return ELSmatchMetrics(precision, recall, f_score, match_num, test_num, gold_num)


def main(args):
  hstream = open(args.hypo_file, 'r')
  gstream = open(args.gold_file, 'r')
  vout = open(args.vout, 'w')

  index = 0

  el_smatch_scores = []
  while True:
    if VERBOSE:
      print("Scoring pair {}".format(index))
    index += 1

    hamr = el_smatch.get_amr_line(hstream)
    gamr = el_smatch.get_amr_line(gstream)
    if gamr == "" and hamr == "":
      break
    if hamr == "":
      print("Error: File 1 has less AMRs than file 2", file=ERROR_LOG) 
      print("Ignoring remaining AMRs", file=ERROR_LOG)
      break
    if gamr == "":
      print("Error: File 2 has less AMRs than file 1", file=ERROR_LOG)
      print("Ignoring remaining AMRs", file=ERROR_LOG)
      break
  
    if VERBOSE:
      print(hamr)
      print(gamr)
    hamr = el_amr.AMR.parse_AMR_line(hamr)
    gamr = el_amr.AMR.parse_AMR_line(gamr)
  
    # Set up the amrs.
    prefix1 = "a"
    prefix2 = "b"
    # Rename node to "a1", "a2", .etc
    hamr.rename_node(prefix1)
    # Renaming node to "b1", "b2", .etc
    gamr.rename_node(prefix2)
  
    # Get the triples for expanded amrs and run el_smatch.
    (instance1, attribute1, relation1) = hamr.get_triples()
    (instance2, attribute2, relation2) = gamr.get_triples()
    mapping, match_num = \
        el_smatch_run(instance1, attribute1, relation1,
                      instance2, attribute2, relation2,
                      prefix1, prefix2, args.iterations)
  
    # Record scores for each mapping.
    cur_el_smatch_scores = \
        score_el_smatch_mapping(mapping, match_num,
            instance1, attribute1, relation1, 
            instance2, attribute2, relation2)
    if VERBOSE:
        print(cur_el_smatch_scores)

    # Store scores for later.
    el_smatch_scores.append(cur_el_smatch_scores)

    # clear the matching triple dictionary for the next AMR pair
    el_smatch.match_triple_dict.clear()
    
  # Print out results. 
  out = open(args.out, 'w')
  doc_scores = calculate_document_score(el_smatch_scores)
  if VERBOSE:
    print(doc_scores, file=vout)
    print(doc_scores)
  else:
    print(" ".join([str(x) for x in doc_scores]))
  vout.close()
  out.write('(')
  out.write('\n'.join(['({})'.format(' '.join([str(v) for v in values])) for values in el_smatch_scores]))
  out.write(')')
  out.write('\n\n')
  out.write('({})'.format(' '.join([str(v) for v in doc_scores])))
  out.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--hypo_file', type=str, required=True, help='Hypothesis EL-AMR file')
  parser.add_argument('--gold_file', type=str, required=True, help='Gold EL-AMR file')
  parser.add_argument('--out', type=str, required=True, help='Output file')
  parser.add_argument('--vout', type=str, required=True, help='Verbose output file')
  parser.add_argument('--iterations', type=int, default=5, help='Number of iterations to run EL-smatch to reduce noise')
  parser.add_argument('--verbose', action='store_true', help='Whether to print out computation results to stdout')
  args, unparsed = parser.parse_known_args()
  VERBOSE = args.verbose
  main(args)


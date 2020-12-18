#!/usr/bin/python
import pickle
import sys
import os
import amr_graph
from amr_graph import *
from re_utils import *

def get_amr_line(input_f):
    """Read the amr file. AMRs are separated by a blank line."""
    cur_amr=[]
    cur_amrs=[]
    has_content=False
    for line in input_f:
      if line[0]=="(" and len(cur_amr)!=0:
         cur_amrs.append(cur_amr)
         cur_amr=[]
      if line.strip()=="":
         if not has_content:
            continue
         else:
            break
      elif line.strip().startswith("#") or line.strip().startswith(";"):
        # omit the comment in the AMR file
        continue
      else:
         has_content=True
         cur_amr.append(delete_pattern(line.strip(), '~e\.[0-9]+(,[0-9]+)*'))
    if len(cur_amr) != 0:
      cur_amrs.append(cur_amr)
    temp = "({})".format(" ".join([" ".join(cur_amr) for cur_amr in cur_amrs]))
    if temp[:2] == "((" and temp[-2:] == "))":
        temp = temp[1:-1]
    if temp == "()":
        temp = ""
    return temp

#Load a list of amr graph objects
def load_amr_graphs(amr_file, ignore_firstnlast=False):

    f = open(amr_file, 'r')
    amr_line = get_amr_line(f)
    graphs = []

    curr_index = 0
    i = 0
    while amr_line and amr_line.strip() != '':
        #print("i: {}".format(i))
        i += 1 
        #print(amr_line)
        #fflush(stdout)
        if ignore_firstnlast:
          amr_line = amr_line.strip()
          amr_line = amr_line[1:len(amr_line) - 1]
        amr_graph = AMRGraph(amr_line)
        graphs.append(amr_graph)
        #if len(graphs) % 5000 == 0:
        #    curr_dump_file = os.path.join(divide_dir, 'graph_%d' % curr_index)
        #    curr_f = open(curr_dump_file, 'wb')
        #    pickle.dump(graphs, curr_f)
        #    curr_f.close()
        #    curr_index += 1
        #    graphs[:] = []
        amr_line = get_amr_line(f)

    return graphs

def to_single_line(amr_file, output_file):
    f = open(amr_file, 'r')
    amr_line = get_amr_line(f)

    curr_index = 0
    with open(output_file, 'w') as wf:
        while amr_line and amr_line != '()':
            print(amr_line, file=wf)
            amr_line = get_amr_line(f)
        wf.close()

def to_original_order(indir, output, token_path):
    old_toks = utils.loadTokens(token_path)
    new_tok_file = os.path.join(indir, "toks")
    new_idx_file = os.path.join(indir, "idxs")
    new_amr_file = os.path.join(indir, "amr")
    new_toks = utils.loadTokens(new_tok_file)
    sent_idxs = [int(line.strip()) for line in open(new_idx_file)]
    amr_lines = [line.strip() for line in open(new_amr_file)]
    assert len(old_toks) == len(new_toks), "%d\n%d\n" % (len(old_toks), len(new_toks))
    ordered_amrs = ["" for _ in range(len(new_toks))]
    for (sent_idx, amr_line) in enumerate(amr_lines):
        orig_sent_idx = sent_idxs[sent_idx]
        try:
            assert " ".join(old_toks[orig_sent_idx]).encode("utf-8") == " ".join(new_toks[sent_idx]).encode("utf-8")
        except:
            print(" ".join(old_toks[orig_sent_idx]))
            print(" ".join(new_toks[sent_idx]))
        ordered_amrs[orig_sent_idx] = amr_line

    with open(output, "w") as wf:
        for (sent_idx, amr_line) in enumerate(ordered_amrs):
            print("%s\n" % amr_line, file=wf)
        wf.close()

if __name__ == "__main__":
    input_file = sys.argv[1]
    tok_file = sys.argv[2]
    input_dir = sys.argv[3]
    output_file = sys.argv[4]
    output_amr = sys.argv[5]
    to_single_line(input_file, output_file)
    to_original_order(input_dir, output_amr, tok_file)


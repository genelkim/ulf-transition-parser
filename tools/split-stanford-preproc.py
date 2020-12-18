"""
Splits up the result of Stanford CoreNLP preprocessing into individual files
for each type of annotation.

This script assumes the annotations done by Stanford CoreNLP includes
  tokenize, pos, ner, and lemma
and is output into json format.
"""

import sys
import os
import json

if len(sys.argv) < 3:
  sys.exit("Usage python split-stanford-preproc.py [input file] [output directory]")

TOKEN_FILE = "token"
POS_FILE = "pos"
NER_FILE = "ner"
LEMMA_FILE = "lemma"

jd = json.loads(open(sys.argv[1], 'r').read())

def format_tokens(s):
  return u" ".join([t["word"] for t in s["tokens"]])

def format_pos(s):
  return u" ".join([t["pos"] for t in s["tokens"]])

def format_lemma(s):
  return u" ".join([t["lemma"] for t in s["tokens"]])

def format_ner(s):
  return u"\n".join([u"{}\t{}".format(t["word"], t["ner"]) for t in s["tokens"]])

tokens = []
poses = []
lemmas = []
ners = []

for s in jd["sentences"]:
  tokens.append(format_tokens(s))
  poses.append(format_pos(s))
  lemmas.append(format_lemma(s))
  ners.append(format_ner(s))

tout = open(os.path.join(sys.argv[2], TOKEN_FILE), 'wb')
tout.write(u"\n".join(tokens).encode("utf-8"))
tout.close()
pout = open(os.path.join(sys.argv[2], POS_FILE), 'wb')
pout.write(u"\n".join(poses).encode("utf-8"))
pout.close()
lout = open(os.path.join(sys.argv[2], LEMMA_FILE), 'wb')
lout.write(u"\n".join(lemmas).encode("utf-8"))
lout.close()
nout = open(os.path.join(sys.argv[2], NER_FILE), 'wb')
nout.write(u"\n\n".join(ners).encode("utf-8"))
nout.close()


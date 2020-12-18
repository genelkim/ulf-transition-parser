#!/usr/bin/python3
import sys, os

sent_index = 0

output_dir = sys.argv[2]
os.system("mkdir -p %s" % output_dir)
token_wf = open(os.path.join(sys.argv[2], 'token'), 'w')
pos_wf = open(os.path.join(sys.argv[2], 'pos'), 'w')
toks = []
poss = []

special_symbols = set()
special_token_map = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}
for line in open(sys.argv[1]):
    fields = line.strip().split()
    if len(fields) < 2: #A new sent
        sent_index += 1
        print((' '.join(toks)), file=token_wf)
        print((' '.join(poss)), file=pos_wf)
        toks = []
        poss = []
        continue

    curr_tok = fields[1].strip()

    for sp in special_token_map:
        if sp in curr_tok:
    #if curr_tok in special_token_map:
            curr_tok = curr_tok.replace(sp, special_token_map[sp])

    toks.append(curr_tok)
    if curr_tok[0] == '-' and curr_tok[-1] == '-':
        if len(curr_tok) > 2:
            special_symbols.add(curr_tok)

    poss.append(fields[4].strip())
    # assert fields[3].strip() == fields[4].strip()

token_wf.close()
pos_wf.close()
print(special_symbols)


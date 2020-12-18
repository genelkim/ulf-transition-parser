import sys
sent_index = 0
orig_f = open(sys.argv[1])
tokenized_f = open(sys.argv[2])
orig_span_f = open(sys.argv[3])
result_wf = open(sys.argv[4], "w")

def match_s(toks, start_index, s):
    n_toks = len(toks)
    for start in range(start_index, n_toks):
        for end in range(start+1, n_toks+1):
            curr_s = (' '.join(toks[start:end])).replace("@ - @", "@-@").replace("@ :@", "@:@").replace("@ / @", "@/@")
            #print curr_s
            if curr_s == s:
                return (start, end)
    return (None, None)

for line in orig_f:
    if line.strip() == "":
        break

    orig_toks = line.strip().split()
    tokenized_toks = tokenized_f.readline().strip().split()

    span_line = orig_span_f.readline().strip()

    new_spans = []
    if span_line != "":
        start_index = 0
        for sp in span_line.split():
            start = int(sp.split('-')[0])
            end = int(sp.split('-')[1])
            old_s = ' '.join(orig_toks[start:end])
            new_start, new_end = match_s(tokenized_toks, start_index, old_s)
            if new_start is None:
                print(old_s)
                print(orig_toks)
                print(tokenized_toks)
                sys.exit(1)
            new_spans.append(('%d-%d' % (new_start, new_end)))
            start_index = new_end
        print(' '.join(new_spans), file=result_wf)
    else:
        print("", file=result_wf)


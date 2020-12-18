"""
Given preprocessed sentences and word-atom alignment frequencies generates a sequence of ULF atoms.
"""

from ulf_align import split_ulf_atom, AnnSents, AnnToken, AnnSent
import argparse
import json

# Generates ULF atoms from words, lemmas, and POS tags and alignment counts.
# Uses pos2ext counts either separately or integrated with token2atomcounts
# TODO: handle multiword -> ULF segment mappings.
def words2atoms(annsent, token2atom, lemma2atom, pos2ext):
  # If this is an ner get contiguous ner and form a |...| TODO: figure out how
  # to recover original text span from the tokens... I think stanford parser
  # preserves the character spans form the original sentence.  ACTUALLY,
  # apparently the NER files have the capitalization preserved tokens!  The
  # original span would still be better, but this can be simple start

  in_ner = False
  cur_ners = []
  atoms = []
  for anntoken in annsent.anntokens:
    token = anntoken.token
    lemma = anntoken.lemma
    pos = anntoken.pos
    ner = anntoken.ner
    if in_ner and ner == "O":
      # Just finished building a named entity.
      atoms.append("|{}|".format(" ".join(cur_ners)))
      cur_ners = []
      in_ner = False
    elif in_ner:
      # Middle of building named enitty.
      cur_ners.append(token)
    elif ner != "O":
      # STarted building a named enitty.
      cur_ners.append(token)
      in_ner = True
    else:
      # Building normal atoms.
      extc = pos2ext[pos] if pos in pos2ext else []
      latc = [(x,y) for x, y in lemma2atom[lemma]] if lemma in lemma2atom else []
      tatc = [(x,y) for x,y in token2atom[token]] if token in token2atom else []

      # TODO preprocess these counts.
      extsum = sum([e[1] for e in extc])
      extmle = { ext : c / extsum for ext, c in extc }
      latsum = sum([e[1] for e in latc])
      latmle = { lat : c / latsum for lat, c in latc }
      tatsum = sum([e[1] for e in tatc])
      tatmle = { tat : c / tatsum for tat, c in tatc }

      # Considered concepts.
      ats = list(set([e[0] for e in latc + tatc]))

      if len(ats) == 0:
        # If there are no available atoms, generate one.
        bestext = "PRO"
        bestscore = -1
        for ext, score in extmle.items():
          if score > bestscore and ext:
            bestscore = score
            bestext = ext
        atoms.append(lemma.upper() + "." + bestext)
      else:
        # Sum the scores of all three with the token having twice the weight of
        # the other two.
        bestat = ats[0]
        bestscore = -1
        for at in ats:
          base, ext = split_ulf_atom(at)
          extscore = 1 / extsum if extsum > 0 else 0
          if ext in extmle:
            extscore = extmle[ext]
          latscore = 1 / latsum if latsum > 0 else 0
          if at in latmle:
            latscore = latmle[at]
          tatscore = 1 / tatsum if tatsum > 0 else 0
          if at in tatmle:
            tatscore = tatmle[at]
          score = tatscore + 0.5 * (latscore + extscore)
          if score > bestscore:
            bestscore = score
            bestat = at
        atoms.append(bestat)
  return atoms


def get_alignment_strings(tokens, atoms):
    aligns = []
    for t, a in zip(tokens, atoms):
        if a == "NONE":
            aligns.append("NONE")
        else:
            aligns.append(t + "||" + a)
    return aligns


def remove_none(raw_atoms, alignments, atom2word):
  """Removes NONE symbols from the atoms and appropriately updates alignments
  and atom2word.
  """
  filtered_atoms = []
  filtered_alignments = []
  filtered_atom2word = []
  for i in range(len(raw_atoms)):
      raw_atom = raw_atoms[i]
      align = alignments[i]
      aw = atom2word[i]
      if raw_atom != "NONE":
        filtered_atoms.append(raw_atom)
        filtered_alignments.append(align)
        filtered_atom2word.append(aw)
  return filtered_atoms, filtered_alignments, filtered_atom2word


def postprocess_atoms(atoms, alignments, atom2word, complex_freq, complex_prefix_size):
  """Cleans up the predicted sequence and inserts some that isn't properly
  handled by prediction sequence. It also appropriately updates alignments and atom2word.

  - Adds COMPLEX according to argument config
  - Adds tense symbols (PRES PAST CF) as well as (PROG PERF) after the
    COMPLEX prefix and every 10 symbols afterwards.
  - Adds macros (SUB, N+PREDS, NP+PREDS, QT-ATTR, *H, *R, etc.) after the
    COMPLEX prefix and every 10 symbols afterwards.
  """
  repeated_symbols = ["SUB", "*H", "REP", "*P", "N+PREDS", "NP+PREDS", "QT-ATTR", "*QT", "K", "KA", "KE", "THT", "ADV-A", "ADV-E", "FQUAN", "NQUAN", "PLUR"]

  new_atoms = ["COMPLEX"] * complex_prefix_size
  new_alignments = ["NONE"] * complex_prefix_size
  new_atom2word = [-1] * complex_prefix_size
  i = 0
  for at, al, aw in zip(atoms, alignments, atom2word):
    if i % complex_freq == 0 and i != 0:
      new_atoms.append("COMPLEX")
      new_alignments.append("NONE")
      new_atom2word.append(-1)
    if i % 10 == 0:
      new_atoms.extend(repeated_symbols)
      new_alignments.extend(["NONE"] * len(repeated_symbols))
      new_atom2word.extend([-1] * len(repeated_symbols))
    new_atoms.append(at)
    new_alignments.append(al)
    new_atom2word.append(aw)
    i += 1
  return new_atoms, new_alignments, new_atom2word



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--annsent_dir", type=str, required=True, help="Path to the annotated data")
  parser.add_argument("--atom_counts", type=str, required=True, help="Path to the JSON file with ULF atom and extension counts.")
  parser.add_argument("--outfile", type=str, required=True, help="Output file path.")
  parser.add_argument("--align_out", type=str, required=True, help="Output file path for alignments.")
  parser.add_argument("--atom2word_out", type=str, required=True, help="Output file path for atom to word mappings.")
  parser.add_argument("--predict_style", type=str, required=True, choices=["raw", "overgen"],
          help="The style of atom generation to use. \"raw\" only generates aligned words. \"overgen\" over-generates un-aligned symbols, so that the parser can choose to ignore the extra symbols.")
  parser.add_argument("--complex_freq", type=int, default=2, help="Every how many words to insert the COMPLEX symbol. Only used if predict_style=\"overgen\".")
  parser.add_argument("--complex_prefix_size", type=int, default=3, help="How many COMPLEX symbols to prefix at the beginning fo the sequence. Only used if predict_style=\"overgen\".")
  parser.add_argument("--remove_none", action="store_true", help="Whether to filter out NONE atoms.")
  args, unparsed_args = parser.parse_known_args()

  annsents = AnnSents(args.annsent_dir)
  atcounts = json.loads(open(args.atom_counts, 'r').read())

  sent_atoms = []
  sent_aligns = []
  sent_atom2word = []
  for annsent in annsents.annsents:
    atoms = words2atoms(
        annsent,
        atcounts["token2atom"],
        atcounts["lemma2atom"],
        atcounts["pos2ext"],
    )
    atom2word = list(range(len(atoms)))
    tokens = [at.token for at in annsent.anntokens]
    alignments = get_alignment_strings(tokens, atoms)
    if args.remove_none:
      atoms, alignments, atom2word = remove_none(atoms, alignments, atom2word)

    if args.predict_style == "overgen":
        atoms, alignments, atom2word = postprocess_atoms(
            atoms,
            alignments,
            atom2word,
            args.complex_freq,
            args.complex_prefix_size,
        )
    sent_atoms.append(atoms)
    sent_aligns.append(alignments)
    sent_atom2word.append(atom2word)

  out = open(args.outfile, "w")
  out.write("\n".join(["\t".join(atoms) for atoms in sent_atoms]))
  out.close()
  with open(args.align_out, "w") as aout:
    aout.write("\n".join(["_#_".join(align) for align in sent_aligns]))
  with open(args.atom2word_out, 'w') as awout:
    json.dump(sent_atom2word, awout)


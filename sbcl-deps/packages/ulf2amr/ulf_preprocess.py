"""
ULF preprocessing code that is simpler to handle in Python than in Lisp. For 
the portion that is handled in Lisp, see ulf-preprocess.lisp

ULF preprocessing:
  1. `\", `\' -> (quotestart-i \"), (quotestart-i \')
     '\", '\' -> (quotestart-o \"), (quotestart-o \')
     `"_", `'_' -> (quote-i "_"), (quote-i |'_'|)
     '"_", ''_' -> (quote-o "_"), (qoute-o |'_'|)
  2. Check that other punctuation is properly escaped.
     
  TODO: lambda expansion? negation/sentence-op lifting? -- probably easier in Lisp.
          
"""

import re
import sys

# Quote regex mapping.
#qregexmap = { \
#  r'`\\"'                   : r'(quotestart-i \\")', \
#  r'`\\\''                  : '(quotestart-i \\\')', \
#  r'\'\\"'                  : r'(quotestart-o \\")', \
#  r'\'\\\''                 : '(quotestart-o \\\')', \
#  r'`"((?:\\.|[^"\\])*)"'     : '(quote-i "{}")', \
#  r'`\'((?:\\.|[^\'\\])*)\''  : '(quote-i |\'{}\'|)', \
#  r'\'"((?:\.|[^"\\])*)"'    : '(quote-o "{}")', \
#  r'\'\'((?:\.|[^\'\\])*)\'' : '(quote-o |\'{}\'|)' \
#}

# Use a list of pairs so that we can control the order that they are applied.
qregexpairs = [
  (r'\\\\"' , r'qtsym'),
  (r'\\"'   , r'qtsym')
]

def quote_pre(s):
  curs = s
  for p, r in qregexpairs:
    while re.search(p, curs):
      res = re.search(p, curs)
      repl = r
      if res.groups():
        repl = r.format(res.groups()[0])
      curs = re.sub(p, repl, curs, 1)
  return curs

escaped_pregexmap = {\
  "Comma"       : r'[^\\],',\
  "Semicolon"   : r'[^\\];',\
  "Period"      : r'[^\\]\.',\
  "Doublequote" : r'[^\\]"',\
  "Singlequote" : r'[^\\]\''
}

not_escaped_pregexmap = {\
  "Exclamation"     : r'\\!',
  "QuestionMark"    : r'\\\?'}

# \. \; \, \" \' should be escaped, but not ! ? 
def check_escape(s):
  passed = True
  message = ""
  for punct, pat in escaped_pregexmap.iteritems():
    if re.search(pat, s):
      passed = False
      ms = re.finditer(pat, s)
      message += punct + " failed because not escaped at [" + ",".join([str(m.start(0)) for m in ms]) + "]"

  for punct, pat in not_escaped_pregexmap.iteritems():
    if re.search(pat, s):
      passed = False
      ms = re.finditer(pat, s)
      message += punct + " failed because escaped at [" + ",".join([str(m.start(0)) for m in ms]) + "]"
  
  if passed:
    message = "All punctuation looks good!"
  return (passed, message)

# Replace names with quotes.
# NB: This preprocessing is only necessary for interacting with Python, e.g. 
# ULFSmatch code.
def name2quote(s):
  return s.replace("|", "\"")

# Adds a space in front of names so Lisp preserves the pipes.
def add_pipe_space(s):
  # TODO: use a version that does the wrap around so we can handle examples 
  # like |.| -> | .|.  Something like : re.sub(r'\|([^\|]\\*)\|', r'| \1|', sent)
  # That one doesn't quite work though when there are multiple...
  return re.sub(r'\|([^.])', r'| \1', s)

# Filter curly brackets {}, unless escaped.
# TODO: change the type system to handle this.  Probably preprocess so:
# {(...)} -> (elided (...)) [RIGHT NOW WE JUST DELETE {}]
# {ref1}.n-- handled in type system since it's lexical.
def bracket_filter(s):
  return re.sub(r'(?<!\\)([{}])', r'', s)

def escape_commas(s):
  return re.sub(r"([^\\]|^),", r"\1\\,", s)

def preproc(s):
  #preproc_fns = [bracket_filter, quote_pre, name2quote, check_escape]
  # TODO: make sure the comma escaping, bracket filtering, etc. are done inside quotes and pipes.
  preproc_fns = [bracket_filter, quote_pre, add_pipe_space, escape_commas]
  res = s
  for fn in preproc_fns:
    res = fn(res)
  return res

def main():
  if len(sys.argv) < 3:
    sys.exit("Usage: python ulf_process.py [input file] [output file]")
  out = file(sys.argv[2], 'w')
  out.write("{}".format(preproc(file(sys.argv[1], 'r').read())))
  out.close()

if __name__ == "__main__":
  main()



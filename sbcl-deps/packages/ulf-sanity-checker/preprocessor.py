"""
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
qregexmap = { \
  r'`\\"'                   : r'(quotestart-i \\")', \
  r'`\\\''                  : '(quotestart-i \\\')', \
  r'\'\\"'                  : r'(quotestart-o \\")', \
  r'\'\\\''                 : '(quotestart-o \\\')', \
  r'`"((?:\\.|[^"\\])*)"'     : '(quote-i "{}")', \
  r'`\'((?:\\.|[^\'\\])*)\''  : '(quote-i |\'{}\'|)', \
  r'\'"((?:\.|[^"\\])*)"'    : '(quote-o "{}")', \
  r'\'\'((?:\.|[^\'\\])*)\'' : '(quote-o |\'{}\'|)' \
}

def quote_pre(s):
  curs = s
  for p, r in qregexmap.iteritems():
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

# Filter curly brackets {}, unless escaped.
# TODO: change the type system to handle this.  Probably preprocess so:
# {(...)} -> (elided (...))
# {ref1}.n-- handled in type system since it's lexical.
def bracket_filter(s):
  return re.sub(r'(?<!\\)([{}])', r'', s)

# Add a space before all ULF names that are in uppercase.
def name_preprocess(s):
  namepat = r"\|[^\|]+\|"
  ms = re.finditer(namepat, s)
  curstr = s
  foundcount = 0
  for m in ms:
    txt = m.group(0)
    if txt.upper() == txt:
      idx = m.start() + 1 + foundcount
      foundcount += 1
      curstr = curstr[:idx] + " " + curstr[idx:]
  return curstr

def preproc(s):
  bracket_filtered = bracket_filter(s)
  quoted = quote_pre(bracket_filtered)
  escape_res = check_escape(quoted)
  name_preprocd = name_preprocess(quoted)
  #print escape_res[1]
  return name_preprocd

def main():
  if len(sys.argv) < 3:
    sys.exit("Usage: preprocessor.py [input file] [output file]")
  out = file(sys.argv[2], 'w')
  out.write("{}".format(preproc(file(sys.argv[1], 'r').read())))
  out.close()

if __name__ == "__main__":
  main()



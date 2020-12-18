"""
Post-processing script for ULF sanity checker to eliminate compilation messages
before the main script output.

Takes a file as input and outputs to stdout.
"""

import sys

SEP = "************************************"

if len(sys.argv) != 2:
  sys.exit("Usage: postprocessor.py [input file]")

text = open(sys.argv[1], 'r').read()

split = text.split(SEP)

print(SEP+SEP.join(split[1:]))


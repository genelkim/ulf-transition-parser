
from preprocessor import *

testlines = file("preproc_testfile", 'r').read().splitlines()

for t in testlines:
  res = quote_pre(t)
  print "Test"
  print t
  print res
  print ""


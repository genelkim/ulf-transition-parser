#!/bin/bash

# Exit on failure
set -e

TEMPFILE=./.temp.py_preprocessed
TEMPFILE2=./.temp.lisp_preprocessed
CURDIR=~/research/ulf2amr/

python ${CURDIR}ulf_preprocess.py ${1} ${TEMPFILE}
${CURDIR}ulf-preprocess-script.cl ${TEMPFILE} > ${TEMPFILE2}
./ulf2amr.cl ${TEMPFILE2} 

rm ${TEMPFILE}
rm ${TEMPFILE2}
rm ${PREPROCESSED}


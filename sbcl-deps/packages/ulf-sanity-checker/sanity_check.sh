#!/bin/bash


CHECKER_DIR="../ulf-sanity-checker"

CURDIR=$(pwd)

TEMPFILE1=$(mktemp temp.preprocessed.XXXXXXXX)
TEMPFILE2=$(mktemp temp.postprocessed.XXXXXXXX)

cd $CHECKER_DIR

python preprocessor.py ${1} ${TEMPFILE1}
./sanity-check.cl ${TEMPFILE1} > ${TEMPFILE2}
python postprocessor.py ${TEMPFILE2}
rm ${TEMPFILE1}
rm ${TEMPFILE2}

cd $CURDIR


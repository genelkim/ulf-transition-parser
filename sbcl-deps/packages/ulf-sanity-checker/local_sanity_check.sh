#!/bin/bash

TEMPFILE1=$(mktemp temp.preprocessed.XXXXXXXX)
TEMPFILE2=$(mktemp temp.postprocessed.XXXXXXXX)

python preprocessor.py ${1} ${TEMPFILE1}
./sanity-check.cl ${TEMPFILE1} > ${TEMPFILE2} 
python postprocessor.py ${TEMPFILE2}
#cat ${TEMPFILE2}
rm ${TEMPFILE1}
rm ${TEMPFILE2}


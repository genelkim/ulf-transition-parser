#!/usr/bin/env bash
#
# Runs some preprocessing from the Stanford parser then calls a python script
# to break out the annotations into individual files.
# Takes at least three arguments:
#   $1: stanford corenlp directory
#   $2: output directory
#   $3: input/output directory
#   $4: input filename

if [ ! $# -ge 1 ]; then
  echo Usage: `basename $0` 'file(s)'
  echo
  exit
fi

scriptdir=`dirname $0`

echo "Running Stanford CoreNLP..."
java -mx20g -cp "$scriptdir/${1}*:" edu.stanford.nlp.pipeline.StanfordCoreNLP \
 -annotators "tokenize,ssplit,pos,lemma,ner" -ssplit.eolonly -file ${3}/${4} -outputFormat json -outputDirectory ${3}

echo "Splitting results into individual files..."
python "$scriptdir/"split-stanford-preproc.py "${3}/${4}.json" ${2}
echo "Deleting immediate CoreNLP output file..."
rm "${3}/${4}.json"

echo "Running Stanford CoreNLP..."
java -mx20g -cp "$scriptdir/${1}*:" edu.stanford.nlp.pipeline.StanfordCoreNLP \
 -annotators "tokenize,ssplit,pos,lemma,ner,depparse" -ssplit.eolonly -file ${3}/${4} -outputFormat conll
mv ${4}.conll ${2}/dep 
# -annotators "tokenize,ssplit,pos,lemma,ner,depparse" -ssplit.eolonly -file ${3}/${4} -outputFormat json -outputDirectory ${3}



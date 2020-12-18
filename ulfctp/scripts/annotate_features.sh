#!/usr/bin/env bash

set -e

# Start a Stanford CoreNLP server before running this script.
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

# The compound file is downloaded from
# https://github.com/ChunchuanLv/AMR_AS_GRAPH_PREDICTION/blob/master/data/joints.txt
touch joints.tmp
#compound_file=data/AMR/amr_2.0_utils/joints.txt
compound_file=joints.tmp
data_dir=$1

python -u -m ulfctp.data.dataset_readers.ulf_parsing.preprocess.feature_annotator \
    ${data_dir}/test.txt ${data_dir}/train.txt ${data_dir}/dev.txt \
    --compound_file ${compound_file}
rm joints.tmp

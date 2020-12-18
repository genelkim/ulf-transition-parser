#!/bin/bash

# Exit on failure, since this is a pipeline...
set -e

# Use the stanford parser to get POS and NER tags and tokenize and lemmatize
CORENLP_VER=stanford-corenlp-full-2018-10-05

# First split up the ULF data into sentences and ULFs.
CACHE_SIZE=2
INPUT_DATA_FILE="initial-ulf-data/ulf-1.0.json"
SET_NAME="ulf-1.0"
BASE_FOLDER=ulfdata/${SET_NAME}
TRAIN_VER="${SET_NAME}-train"
DEV_VER="${SET_NAME}-dev"
TEST_VER="${SET_NAME}-test"
TRAIN_DATA=${BASE_FOLDER}/train
DEV_DATA=${BASE_FOLDER}/dev
TEST_DATA=${BASE_FOLDER}/test
TRAIN_CONLL=${TRAIN_DATA}/conll
DEV_CONLL=${DEV_DATA}/conll
TEST_CONLL=${TEST_DATA}/conll

## Move the original data to the processing dir.
#echo "Splitting and copying original data to working directory..."
#python initial-ulf-data/split-data.py \
#    --input ${INPUT_DATA_FILE} \
#    --trainpath ${TRAIN_DATA} \
#    --testpath ${TEST_DATA} \
#    --devpath ${DEV_DATA}
#
## the sentences.
#TAGGER_DIR=./tools
## Args: [corenlp] [output dir] [input/output directory] [input file]
## Stanford parser
#echo "Running Stanford parser..."
#${TAGGER_DIR}/stanford-preprocess.sh ${CORENLP_VER}/ ${TRAIN_DATA} ${TRAIN_DATA} raw
#${TAGGER_DIR}/stanford-preprocess.sh ${CORENLP_VER}/ ${DEV_DATA} ${DEV_DATA} raw
#${TAGGER_DIR}/stanford-preprocess.sh ${CORENLP_VER}/ ${TEST_DATA} ${TEST_DATA} raw

# Assume for now that ${TRAIN_DATA}/amr contains that AMR formatted ULF formulas.
echo "ULF Align..."
python3 data_processing/ulf_align.py \
    --data_dir ${TRAIN_DATA} \
    --ulfamr_file ${TRAIN_DATA}/amr \
    --output ${TRAIN_DATA}/alignment.amr \
    --amrtype xiaochang \
    --merge_nodes \
    --gen_symbol_file data_processing/gen_symbols
python3 data_processing/ulf_align.py \
    --data_dir ${DEV_DATA} \
    --ulfamr_file ${DEV_DATA}/amr \
    --output ${DEV_DATA}/alignment.amr \
    --amrtype xiaochang \
    --merge_nodes \
    --gen_symbol_file data_processing/gen_symbols
python3 data_processing/ulf_align.py \
    --data_dir ${TEST_DATA} \
    --ulfamr_file ${TEST_DATA}/amr \
    --output ${TEST_DATA}/alignment.amr \
    --amrtype xiaochang \
    --merge_nodes \
    --gen_symbol_file data_processing/gen_symbols

# Generate the CoNLL
# Reformatting
echo "Reformatting..."
python3 data_processing/prepareTokens.py \
  --task reformat \
  --data_dir ${TRAIN_DATA} \
  --use_lemma \
  --run_dir ${TRAIN_CONLL} \
  --stats_dir ${TRAIN_CONLL}/stats \
  --conll_file ${TRAIN_CONLL}/ulfamr_conll \
  --table_dir ${TRAIN_DATA}/tables
cp ${TRAIN_DATA}/dep ${TRAIN_CONLL}/dep
python3 data_processing/prepareTokens.py \
  --task reformat \
  --data_dir ${DEV_DATA} \
  --use_lemma \
  --run_dir ${DEV_CONLL} \
  --stats_dir ${DEV_CONLL}/stats \
  --conll_file ${DEV_CONLL}/ulfamr_conll \
  --table_dir ${DEV_DATA}/tables
cp ${DEV_DATA}/dep ${DEV_CONLL}/dep
python3 data_processing/prepareTokens.py \
  --task reformat \
  --data_dir ${TEST_DATA} \
  --use_lemma \
  --run_dir ${TEST_CONLL} \
  --stats_dir ${TEST_CONLL}/stats \
  --conll_file ${TEST_CONLL}/ulfamr_conll \
  --table_dir ${TEST_DATA}/tables
cp ${TEST_DATA}/dep ${TEST_CONLL}/dep

# Generate predicted alignments.
# Only use the training set for generating alignments. If we're just using an
# example, then it doesn't matter.
echo "Generate atom counts..."
python3 data_processing/ulf_atom_counts.py \
  --annsent_dir ${TRAIN_DATA} \
  --alignment_file ${TRAIN_DATA}/conll/alignment \
  --outfile ulf_atom_counts/${TRAIN_VER}.ua_count
python3 data_processing/predict_ulf_atoms.py \
  --annsent_dir ${TRAIN_DATA} \
  --atom_counts ulf_atom_counts/${TRAIN_VER}.ua_count \
  --outfile ulf_atom_counts/${TRAIN_VER}.predicted_symbols \
  --align_out ulf_atom_counts/${TRAIN_VER}.predicted_alignments \
  --atom2word_out ulf_atom_counts/${TRAIN_VER}.predicted_atom2word \
  --predict_style raw \
  --remove_none
python3 data_processing/predict_ulf_atoms.py \
  --annsent_dir ${DEV_DATA} \
  --atom_counts ulf_atom_counts/${TRAIN_VER}.ua_count \
  --outfile ulf_atom_counts/${DEV_VER}.predicted_symbols \
  --align_out ulf_atom_counts/${DEV_VER}.predicted_alignments \
  --atom2word_out ulf_atom_counts/${DEV_VER}.predicted_atom2word \
  --predict_style raw \
  --remove_none
python3 data_processing/predict_ulf_atoms.py \
  --annsent_dir ${TEST_DATA} \
  --atom_counts ulf_atom_counts/${TRAIN_VER}.ua_count \
  --outfile ulf_atom_counts/${TEST_VER}.predicted_symbols \
  --align_out ulf_atom_counts/${TEST_VER}.predicted_alignments \
  --atom2word_out ulf_atom_counts/${TEST_VER}.predicted_atom2word \
  --predict_style raw \
  --remove_none

# Generate oracle
echo "Generate oracle..."
TRAIN_ORACLE_DIR=ulfdata/oracle/${TRAIN_VER}_cache${CACHE_SIZE}
DEV_ORACLE_DIR=ulfdata/oracle/${DEV_VER}_cache${CACHE_SIZE}
TEST_ORACLE_DIR=ulfdata/oracle/${TEST_VER}_cache${CACHE_SIZE}
python3 -m oracle.oracle \
  --data_dir ${DEV_CONLL} \
  --output_dir ${DEV_ORACLE_DIR} \
  --cache_size ${CACHE_SIZE} \
  --ulf \
  --reserve_gen_method promote \
  --promote_symbol_file data_processing/promote_symbols \
  --inseq_symbol_file data_processing/inseq_symbols \
  --unreserve_gen_method word \
  --focus_method cache \
  --buffer_offset 1
python3 -m oracle.oracle \
  --data_dir ${TRAIN_CONLL} \
  --output_dir ${TRAIN_ORACLE_DIR} \
  --cache_size ${CACHE_SIZE} \
  --ulf \
  --reserve_gen_method promote \
  --promote_symbol_file data_processing/promote_symbols \
  --inseq_symbol_file data_processing/inseq_symbols \
  --unreserve_gen_method word \
  --focus_method cache \
  --buffer_offset 1
python3 -m oracle.oracle \
  --data_dir ${TEST_CONLL} \
  --output_dir ${TEST_ORACLE_DIR} \
  --cache_size ${CACHE_SIZE} \
  --ulf \
  --reserve_gen_method promote \
  --promote_symbol_file data_processing/promote_symbols \
  --inseq_symbol_file data_processing/inseq_symbols \
  --unreserve_gen_method word \
  --focus_method cache \
  --buffer_offset 1

cp ${TRAIN_DATA}/dep ${TRAIN_ORACLE_DIR}/dep
cp ${TRAIN_DATA}/token ${TRAIN_ORACLE_DIR}/token
cp ${TRAIN_DATA}/pos ${TRAIN_ORACLE_DIR}/pos
cp ${TRAIN_DATA}/pos ${TRAIN_ORACLE_DIR}/ner
cp ${TRAIN_DATA}/lemma ${TRAIN_ORACLE_DIR}/lemma
cp ulf_atom_counts/${TRAIN_VER}.predicted_symbols ${TRAIN_ORACLE_DIR}/symbol.pred
cp ulf_atom_counts/${TRAIN_VER}.predicted_alignments ${TRAIN_ORACLE_DIR}/alignments.pred
cp ulf_atom_counts/${TRAIN_VER}.predicted_atom2word ${TRAIN_ORACLE_DIR}/atom2word.pred
cp ${DEV_DATA}/dep ${DEV_ORACLE_DIR}/dep
cp ${DEV_DATA}/token ${DEV_ORACLE_DIR}/token
cp ${DEV_DATA}/pos ${DEV_ORACLE_DIR}/pos
cp ${DEV_DATA}/pos ${DEV_ORACLE_DIR}/ner
cp ${DEV_DATA}/lemma ${DEV_ORACLE_DIR}/lemma
cp ulf_atom_counts/${DEV_VER}.predicted_symbols ${DEV_ORACLE_DIR}/symbol.pred
cp ulf_atom_counts/${DEV_VER}.predicted_alignments ${DEV_ORACLE_DIR}/alignments.pred
cp ulf_atom_counts/${DEV_VER}.predicted_atom2word ${DEV_ORACLE_DIR}/atom2word.pred
cp ${TEST_DATA}/dep ${TEST_ORACLE_DIR}/dep
cp ${TEST_DATA}/token ${TEST_ORACLE_DIR}/token
cp ${TEST_DATA}/pos ${TEST_ORACLE_DIR}/pos
cp ${TEST_DATA}/pos ${TEST_ORACLE_DIR}/ner
cp ${TEST_DATA}/lemma ${TEST_ORACLE_DIR}/lemma
cp ulf_atom_counts/${TEST_VER}.predicted_symbols ${TEST_ORACLE_DIR}/symbol.pred
cp ulf_atom_counts/${TEST_VER}.predicted_alignments ${TEST_ORACLE_DIR}/alignments.pred
cp ulf_atom_counts/${TEST_VER}.predicted_atom2word ${TEST_ORACLE_DIR}/atom2word.pred

cp ${TRAIN_DATA}/ulf.amr-format ${BASE_FOLDER}/train.txt
cp ${DEV_DATA}/ulf.amr-format ${BASE_FOLDER}/dev.txt
cp ${TEST_DATA}/ulf.amr-format ${BASE_FOLDER}/test.txt
cp ${TRAIN_ORACLE_DIR}/oracle_examples.json ${BASE_FOLDER}/train/
cp ${DEV_ORACLE_DIR}/oracle_examples.json ${BASE_FOLDER}/dev/
cp ${TEST_ORACLE_DIR}/oracle_examples.json ${BASE_FOLDER}/test/
cp ${TRAIN_ORACLE_DIR}/*.pred ${TRAIN_DATA}/
cp ${DEV_ORACLE_DIR}/*.pred ${DEV_DATA}/
cp ${TEST_ORACLE_DIR}/*.pred ${TEST_DATA}/


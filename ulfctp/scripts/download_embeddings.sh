#!/usr/bin/env bash

set -e

echo "Downloading artifacts."
mkdir -p data/bert-base-cased
curl -O https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz
tar -xzvf bert-base-cased.tar.gz -C data/bert-base-cased
curl -o data/bert-base-cased/bert-base-cased-vocab.txt \
    https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt
rm bert-base-cased.tar.gz

mkdir -p data/glove
curl -L -o data/glove/glove.840B.300d.zip \
    http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip

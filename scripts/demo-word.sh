#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

if [ $# != 2 ]; then
  echo "USAGE: $0 <FILE> DIM"
  exit 1;
fi

TEXT_DATA=$1
DIM=$2
VECTOR_DATA="${TEXT_DATA}.word2vec.${DIM}.vec"
VOCAB_FILE="${TEXT_DATA}.word2vec.${DIM}.vec.vocab"

pushd ${SRC_DIR} && make; popd

if [ ! -e $VECTOR_DATA ]; then

  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -train $TEXT_DATA -save-vocab $VOCAB_FILE -output $VECTOR_DATA -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1

fi

echo -----------------------------------------------------------------------------------------------------
echo save the file: $VECTOR_DATA

#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

if [ $# != 3 ]; then
    echo "Usage: $0 <FILE> SENSE DIM "
    exit 1;
fi

TEXT_DATA=$1
SENSE_K=$2
DIM=$3
VECTOR_DATA="${TEXT_DATA}.sense${SENSE_K}.sense2vec_mssg.${DIM}.vec"

neg=10
hs=0

echo $VECTOR_DATA


pushd ${SRC_DIR} && make ; popd
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/sense2vec_mssg -train $TEXT_DATA -output $VECTOR_DATA -size $DIM -window 5 -negative $neg -hs $hs -sample 1e-3 -threads 20 -binary 1 -sense $SENSE_K -min-count 50

echo save to $VECTOR_DATA
echo -----------------------------------------------------------------------------------------------------

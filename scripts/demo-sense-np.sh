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

neg=10
hs=0
lambda_sim=-0.5

VECTOR_DATA="${TEXT_DATA}.sense2vec_np.${DIM}.vec"

pushd ${SRC_DIR} && make sense2vec_np; popd

#if [ ! -e $VECTOR_DATA ]; then

  if [ ! -e $TEXT_DATA ]; then
    echo "the file is not exists."
    exit 0
  fi
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/sense2vec_np -train $TEXT_DATA -output $VECTOR_DATA -size $DIM -window 5 -negative $neg -hs $hs -lambda-sim $lambda_sim -sample 1e-3 -threads 20 -binary 1 -min-count 50

##fi

echo save to $VECTOR_DATA
echo -----------------------------------------------------------------------------------------------------

#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/text8
CLASSES_DATA=$DATA_DIR/classes.txt

pushd ${SRC_DIR} && make; popd


if [ ! -e $CLASSES_DATA ]; then

  if [ ! -e $TEXT_DATA ]; then
    wget http://mattmahoney.net/dc/text8.zip -O $DATA_DIR/text8.gz
    gzip -d $DATA_DIR/text8.gz -f
  fi
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1

fi

sort $CLASSES_DATA -k 2 -n > $DATA_DIR/classes.sorted.txt
echo The word classes were saved to file $DATA_DIR/classes.sorted.txt

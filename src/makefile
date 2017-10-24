SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

all: word2vec word2phrase distance word-analogy compute-accuracy  sense2vec sense2vec_np sense2vec_mssg

word2vec : word2vec.c
	$(CC) word2vec.c -o ${BIN_DIR}/word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o ${BIN_DIR}/word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o ${BIN_DIR}/distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o ${BIN_DIR}/word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o ${BIN_DIR}/compute-accuracy $(CFLAGS)
	chmod +x ${SCRIPTS_DIR}/*.sh

# multi sense scripts

sense2vec: sense2vec.c
	$(CC) sense2vec.c -o ${BIN_DIR}/sense2vec $(CFLAGS)

sense2vec_np: sense2vec_np.c
	$(CC) sense2vec_np.c -o ${BIN_DIR}/sense2vec_np $(CFLAGS)

sense2vec_mssg: sense2vec_mssg.c
	$(CC) sense2vec_mssg.c -o ${BIN_DIR}/sense2vec_mssg $(CFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf word2vec  sense2vec sense2vec_mssg  word2phrase distance word-analogy sense2vec_np  compute-accuracy; popd

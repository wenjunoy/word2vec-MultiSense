//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_STOP_WORDS 113

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
  char pos; //stopword and word POS.pos=0 ,uk; pos=1,stopword
};


char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char word2vec_file[MAX_STRING];
char stop_word_file[MAX_STRING];
char syn0_file[MAX_STRING], syn1_file[MAX_STRING], syn1neg_file[MAX_STRING], syn0sense_file[MAX_STRING]; //保存参数到文件中

struct vocab_word *vocab;
int init_sense = -1; // >0 init 0, <0 init to random
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
// multi sense ,K
int sense_K = 3;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0, max_similar = -1.0;
real *syn0, *syn1, *syn1neg, *expTable, *syn0sense, *syn0mu;
int *sense_num;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

char *stop_words;



void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab[vocab_size].pos = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void DestroyVocab() {
  int a;

  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      free(vocab[a].word);
    }
    if (vocab[a].code != NULL) {
      free(vocab[a].code);
    }
    if (vocab[a].point != NULL) {
      free(vocab[a].point);
    }
  }
  free(vocab[vocab_size].word);
  free(vocab);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 1; a < size; a++) { // Skip </s>
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[a].word);
      vocab[a].word = NULL;
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn,&c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

//load the stop_words
int ReadStopWords() {
  /* code */
  int a;
  long long word_index;
  FILE *fi;
  fi =fopen(stop_word_file, "rb");
  if (fi == NULL){
    printf("The stop word file can not found!\n" );
    return -1;
  }
  stop_words = (char *)malloc((MAX_STOP_WORDS+1) * MAX_STRING *sizeof(char));
  for (a = 0; a < MAX_STOP_WORDS; a ++) {
    fscanf(fi, "%s\n", &stop_words[a * MAX_STRING]);
    //printf("%s\n", &stop_words[a*MAX_STRING]);
    word_index = SearchVocab(&stop_words[a * MAX_STRING]);
    //printf("%d\n", word_index);
    if (word_index > -1) {
      vocab[word_index].pos = 1;
    }
  }
  return 1;
}

int InitWord2Vec() {
  long long words, size, a, b, word_index;
  char ch;
  float *M;
  char *word;
  FILE *f = fopen(word2vec_file, "rb");
  printf("load the word vector file, init word2vec with input file: %s\n", word2vec_file);
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  if (size != layer1_size) {
    printf("size= %lld, layer1_size= %lld\n", size, layer1_size);
    printf("warning: the intput word vector size is not same to the layer1_size!\n");
    return -1;
  }
  word = (char *)malloc((long long)words * MAX_STRING * sizeof(char));
  M = (float *)malloc((long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &word[b*MAX_STRING], &ch);
    //printf("%s\n", &word[b*MAX_STRING]);
    for (a = 0; a < size; a++) fread(&M[a], sizeof(float), 1, f);
    word_index = SearchVocab(&word[b*MAX_STRING]);
    //printf("%d,", word_index);
    if (word_index > -1){
        for (a = 0; a < layer1_size; a++) syn0[word_index * layer1_size+ a] = M[a];
    }
  }
  if (M != NULL) free(M);
  if (word != NULL) free(word);

  fclose(f);
  return 1;
}


// save the word vectors.
int SaveSyn0() {
  long long b, a;
  FILE *fo;
  strcpy(syn0_file, save_vocab_file);
  strcat(syn0_file, ".syn0");
  fo= fopen(syn0_file, "wb");
  if (fo == NULL) {
    printf("the save syn0 file can not open!\n");
    return -1;
  }
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    fprintf(fo, "\n");
  }
  fclose(fo);
  return 1;
}


// save the sense vectors
int SaveSyn0sense() {
  long long b, a, k;
  FILE *fo;
  char str[MAX_STRING];
  strcpy(syn0sense_file, save_vocab_file);
  strcat(syn0sense_file, ".syn0sense");
  fo= fopen(syn0sense_file, "wb");
  if (fo == NULL) {
    printf("the save syn0sense file can not open!\n");
    return -1;
  }
  fprintf(fo, "%lld %lld\n", vocab_size * sense_K, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    for (k = 0;k < sense_K; k++){
      sprintf(str, "%lld", k);
      if (vocab[a].word != NULL) {
       fprintf(fo, "%s_%s ", vocab[a].word, str);
      }
      for (b = 0; b < layer1_size; b++) fwrite(&syn0sense[(a*sense_K+k) * layer1_size + b], sizeof(real), 1, fo);
      fprintf(fo, "\n");
    }
  }
  fclose(fo);
  return 1;
}

int SaveSyn1() {
  long long b, a;
  FILE *fo;
  if (!hs) {
    printf("The syn1 is not exists.\n");
    return -1;
  }
  strcpy(syn1_file, save_vocab_file);
  strcat(syn1_file, ".syn1");
  fo= fopen(syn1_file, "wb");
  if (fo == NULL) {
    printf("the save syn1 file can not open!\n");
    return -1;
  }
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    for (b = 0; b < layer1_size; b++) fwrite(&syn1[a * layer1_size + b], sizeof(real), 1, fo);
    fprintf(fo, "\n");
  }
  fclose(fo);
  return 1;
}

int SaveSyn1neg() {
  long long b, a;
  FILE *fo;
  if (negative <= 0) {
    printf("the syn1neg is not exists.\n");
    return -1;
  }
  strcpy(syn1neg_file, save_vocab_file);
  strcat(syn1neg_file, ".syn1neg");
  fo= fopen(syn1neg_file, "wb");
  if (fo == NULL) {
    printf("the save syn1neg file can not open!\n");
    return -1;
  }
  //printf("%lld\n", vocab_size);
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo);
    fprintf(fo, "\n");
  }
  fclose(fo);
  return 1;
}

void InitNet() {
  long long a, b, k;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&syn0sense, 128, (long long)vocab_size* sense_K* layer1_size * sizeof(real));
  if (syn0sense == NULL) {printf("Memory allocation failed\n"); exit(1);}

  a = posix_memalign((void **)&syn0mu, 128, (long long)vocab_size* sense_K* layer1_size * sizeof(real));
  if (syn0mu == NULL) {printf("Memory allocation failed\n"); exit(1);}

  a = posix_memalign((void **)&sense_num, 128, (long long)vocab_size* sense_K * sizeof(int));
  if (sense_num == NULL) {printf("Memory allocation failed\n"); exit(1);}

  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1neg[a * layer1_size + b] = 0;
  }
  // random init word vector
  for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
    syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  // using skip-gram word vectors initial
  if (word2vec_file[0] != 0) InitWord2Vec();
  // init the sense vetor, random init
  if (init_sense > 0) {
    printf("\nInit the sense vector with 0\n");
    for (b = 0; b < layer1_size; b++) for (a=0; a< vocab_size;a++) for(k = 0 ;k <sense_K;k++)
      syn0sense[(a*sense_K+k)*layer1_size +b] = 0.0;
  }else {
    printf("\nInit the sense vectors with randomly.\n");
    for (b = 0; b < layer1_size; b++) for (a=0; a< vocab_size;a++) for(k = 0 ;k <sense_K;k++)
      syn0sense[(a*sense_K+k)*layer1_size +b] = (rand() / (real)RAND_MAX -0.5) / layer1_size;
  }

  printf("\nInit the cluster center with 0\n");
  for (b = 0; b < layer1_size; b++) for (a=0; a< vocab_size;a++) for(k = 0 ;k <sense_K;k++)
    syn0mu[(a*sense_K+k)*layer1_size +b] = 0.0;

  for (a=0; a< vocab_size;a++) for(k = 0 ;k <sense_K;k++) sense_num[a*sense_K+k] = 0;
  CreateBinaryTree();
}

void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn0sense != NULL){
    free(syn0sense);
  }
  if (sense_num != NULL) {
    free(sense_num);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l1_s, ll_sk, l2, c, target, label;
  long long k, window_words_num, arg_max;
  unsigned long long next_random = (long long)id;
  real max;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  real *context_vec = (real*)calloc(layer1_size, sizeof(real));
  real *similarity = (real*)calloc(sense_K, sizeof(real));

  FILE *fi = fopen(train_file, "rb");
  if (fi == NULL) {
    fprintf(stderr, "no such file or directory: %s", train_file);
    exit(1);
  }
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi)) break;
    if (word_count > train_words / num_threads) break;
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    for (c = 0; c < layer1_size; c++) context_vec[c] = 0;

    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;



    //train skip-gram，只有skip-gram模型
    window_words_num = 0;
    for (a=b; a < window * 2 + 1-b;a++) if (a!=window) {
      c = sentence_position - window + a;
      if (c < 0) continue;
      if (c >= sentence_length) continue;
      last_word = sen[c];
      if (last_word == -1) continue;
      if (vocab[last_word].pos == 1) continue; //not consider the stop words

      //选择那些与预测单词相近的单词作为判断选择标准
      //if (similar > max_similar) {
        window_words_num += 1;
        for (c = 0; c < layer1_size; c++) context_vec[c] += syn0[c + last_word * layer1_size];
      //}
    }
    //取平均值和归一化向量
    if (window_words_num > 1){
      for (c = 0; c < layer1_size; c++) context_vec[c] /= window_words_num;
    }

    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      c = sentence_position - window + a;
      if (c < 0) continue;
      if (c >= sentence_length) continue;
      last_word = sen[c];
      if (last_word == -1) continue;
      for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
      for (c = 0; c < sense_K; c++) similarity[c] = 0;
      arg_max = -1;
      max = -10000000.0;
      l1 = last_word * layer1_size;

      //找到相似度最大的那个单词意思,context_vec 与syn0sense对比
      for (k = 0;k < sense_K;k ++) {
        ll_sk = (last_word * sense_K +k) *layer1_size;
        for (c = 0;c < layer1_size; c++) {
          // similarity[k] += context_vec[c] * syn0sense[ll_sk + c];
          similarity[k] += context_vec[c] * syn0mu[ll_sk + c];
        }
        if (similarity[k] > max){
          arg_max = k;  max = similarity[k];
        }
      }
      long long l1_max = last_word * sense_K + arg_max;
      l1_s = l1_max * layer1_size;
      // update the cluster center
      for (c =0 ;c < layer1_size; c++){
        syn0mu[l1_s + c] = (syn0mu[l1_s + c] * sense_num[l1_max] + context_vec[c]) / (sense_num[l1_max] + 1);
      }
      sense_num[l1_max] += 1;


      // HIERARCHICAL SOFTMAX, unuseful
      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output
        for (c = 0; c < layer1_size; c++) f += syn0sense[c + l1_s] * syn1[c + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
        // Learn weights hidden -> output
        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0sense[c + l1_s];
      }


      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        // for (c = 0; c < layer1_size; c++) f += syn0sense[c + l1_s] * syn1neg[c + l2];
        for (c = 0; c < layer1_size; c++) f += syn0sense[c + l1_s] * syn0[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        // for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
        // for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0sense[c + l1_s];
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn0[c + l2];
        for (c = 0; c < layer1_size; c++) syn0[c + l2] += g * syn0sense[c + l1_s];
      }
      // Learn weights input -> hidden
      // for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      for (c = 0; c < layer1_size; c++) syn0sense[c +l1_s] += neu1e[c];
    }

    //end skip-gram model

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d, k;
  FILE *fo, *fo_sense;
  FILE *f_num;
  char str[25];
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (stop_word_file[0] != 0) ReadStopWords();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  // save sense vector and word sense frequency
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    exit(1);
  }
  if (classes == 0) {
    fo_sense = fopen(strcat(output_file, ".sense"), "wb");
    f_num = fopen(strcat(output_file, ".num"), "wb");

    if (fo_sense == NULL || f_num == NULL) {
      fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
      exit(1);
    }
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      if (vocab[a].word != NULL) {
        fprintf(fo, "%s ", vocab[a].word);
      }
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
    // save the word sense vectors
    fprintf(fo_sense, "%lld %lld\n", vocab_size * sense_K, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      for (k = 0;k < sense_K; k++){
        sprintf(str, "%ld", k);
        if (vocab[a].word != NULL) {
         fprintf(fo_sense, "%s_%s ", vocab[a].word, str);
        }
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0sense[(a*sense_K+k) * layer1_size + b], sizeof(real), 1, fo_sense);
        else for (b = 0; b < layer1_size; b++) fprintf(fo_sense, "%lf ", syn0sense[(a*sense_K+k) * layer1_size + b]);
        fprintf(fo_sense, "\n");
      }
    }
    //save the word sense apear number
    for (a = 0; a < vocab_size; a++) {
      for (k = 0;k < sense_K; k++){
        sprintf(str, "%ld", k);
        if (vocab[a].word != NULL) {
         fprintf(f_num, "%s_%s\t\t", vocab[a].word, str);
        }
        fprintf(f_num, "%d\n", sense_num[a*sense_K+k]);
      }
    }

    if (save_vocab_file[0] != 0) {
      SaveSyn0();
      SaveSyn0sense();
      if (hs) SaveSyn1();
      if (negative > 0) SaveSyn1neg();
    }
    fclose(fo_sense);
    fclose(f_num);


  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    if (centcn == NULL) {
      fprintf(stderr, "cannot allocate memory for centcn\n");
      exit(1);
    }
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
  free(table);
  free(pt);
  DestroyVocab();
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t\tUse the continuous back of words model; default is 0 (skip-gram model)\n");

    printf("\t-sense <int>\n");
    printf("\t\tMulti sense number for every word; default is 3 (sense_k = 3)\n");
    printf("\t-init-sense0 <int>\n");
    printf("\t\tif >0 sense vector init with 0,else <0 sense vector init randomly\n");
    printf("\t-wordvec <file>\n");
    printf("\t\twordvec file, use this wordvec init the word vector.\n");

    printf("\t-stopwords <file>\n");
    printf("\t\tstop words file, if you want to delete the stop words in train file\n");

    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -sense 5\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  stop_word_file[0] = 0;
  word2vec_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  
  if ((i = ArgPos((char *)"-sense", argc, argv)) > 0) sense_K = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-init-sense0", argc, argv)) > 0) init_sense = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-wordvec", argc, argv)) > 0) strcpy(word2vec_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-stopwords", argc, argv)) > 0) strcpy(stop_word_file, argv[i+1]);

  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  DestroyNet();
  free(vocab_hash);
  free(expTable);
  return 0;
}

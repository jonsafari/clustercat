#ifndef INCLUDE_CLUSTERCAT_HEADER
#define INCLUDE_CLUSTERCAT_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>			// log(), exp(), pow()
#include <libgen.h>			// basename()
#include <limits.h>			// USHRT_MAX, UINT_MAX
#include <errno.h>
#include "clustercat-data.h"

// Defaults
#define PRIMARY_SEP_CHAR     '\t'
#define PRIMARY_SEP_STRING   "\t"
#define SECONDARY_SEP_CHAR   ' '
#define SECONDARY_SEP_STRING " "
#define TOK_CHARS            " \t\n"
#define UNKNOWN_WORD_CLASS   "UnK"
#define UNKNOWN_WORD         "UnK"
// Number of characters to read-in for each line
#define BUFLEN 8192
#define STDIN_SENT_MAX_CHARS 140000
#define STDIN_SENT_MAX_WORDS 4096
#define MAX_HIST_LEN 15
#define MAX_WORD_LEN 255
#define EULER 2.71828182845904523536

typedef unsigned short sentlen_t; // Number of words in a sentence
#define SENT_LEN_MAX USHRT_MAX

enum class_algos {EXCHANGE, BROWN};

typedef struct {
	sentlen_t length;
	char **sent;
	char **class_sent;
	short word_lengths[STDIN_SENT_MAX_WORDS];
	short class_lengths[STDIN_SENT_MAX_WORDS];
	unsigned int sent_counts[STDIN_SENT_MAX_WORDS];
	float ngram_probs[STDIN_SENT_MAX_WORDS];
} struct_sent_info;

typedef struct {
	char *file_name;
	unsigned long token_count;
	unsigned long line_count;
	unsigned int  type_count;
	unsigned char class_order;
	unsigned char ngram_order : 5;
} struct_model_metadata;

void increment_ngram(struct_map **ngram_map, char * restrict sent[const], const short * restrict word_lengths, short start_position, const sentlen_t i);
unsigned long process_sents_in_buffer(char * restrict sent_buffer[], const long num_sents_in_buffer);
unsigned long process_sent(char * restrict sent_str);
void tokenize_sent(char * restrict sent_str, struct_sent_info *sent_info);

char *argv_0_basename; // Allow for global access to filename

struct cmd_args {
	long           max_sents_in_buffer;
	unsigned short num_threads : 10;
	char           verbose : 3;    // Negative values increasingly suppress normal output
	unsigned char  class_algo : 2; // enum class_algos
};

#endif // INCLUDE_HEADER

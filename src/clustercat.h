#ifndef INCLUDE_CLUSTERCAT_HEADER
#define INCLUDE_CLUSTERCAT_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>			// log(), exp(), pow()
#include <libgen.h>			// basename()
#include <limits.h>			// USHRT_MAX, UINT_MAX
#include <errno.h>

// Defaults
#define PRIMARY_SEP_CHAR     '\t'
#define PRIMARY_SEP_STRING   "\t"
#define SECONDARY_SEP_CHAR   ' '
#define SECONDARY_SEP_STRING " "
#define TOK_CHARS            " \t\n"
#define UNKNOWN_WORD_CLASS   0
#define UNKNOWN_WORD         "UnK"
// Number of characters to read-in for each line
#define BUFLEN 8192
#define STDIN_SENT_MAX_CHARS 50000
#define STDIN_SENT_MAX_WORDS 1024
#define MAX_HIST_LEN 15
#define MAX_WORD_LEN 255
#define EULER 2.71828182845904523536

typedef unsigned short sentlen_t; // Number of words in a sentence
typedef unsigned short wclass_t; // Number of possible word classes
#define SENT_LEN_MAX USHRT_MAX

enum class_algos {EXCHANGE, BROWN};

#include "clustercat-data.h" // bad. chicken-and-egg typedef deps

typedef struct {
	char **sent;
	unsigned int class_sent[STDIN_SENT_MAX_WORDS];
	unsigned int sent_counts[STDIN_SENT_MAX_WORDS];
	short word_lengths[STDIN_SENT_MAX_WORDS];
	sentlen_t length;
} struct_sent_info;

typedef struct {
	char *file_name;
	unsigned long token_count;
	unsigned long line_count;
	unsigned int  type_count;
	unsigned char class_order;
	unsigned char ngram_order : 5;
} struct_model_metadata;

char *argv_0_basename; // Allow for global access to filename

struct cmd_args {
	unsigned long  max_sents_in_buffer;
	char*          dev_file;
	wclass_t       num_classes;
	unsigned short tune_cycles : 10;
	unsigned char  ngram_order : 6;
	unsigned short num_threads : 10;
	unsigned short min_count : 9;
	char           verbose : 3;     // Negative values increasingly suppress normal output
	unsigned char  class_algo : 2;  // enum class_algos
};

void increment_ngram(struct_map **ngram_map, char * restrict sent[const], const short * restrict word_lengths, short start_position, const sentlen_t i);
unsigned long process_sents_in_buffer(char * restrict sent_buffer[], const long num_sents_in_buffer);
unsigned long process_sent(char * restrict sent_str);
void tokenize_sent(char * restrict sent_str, struct_sent_info *sent_info);
void init_clusters(const struct cmd_args cmd_args, unsigned long vocab_size, char **unique_words, struct_map_word_class **word2class_map);
void cluster(const struct cmd_args cmd_args, char * restrict sent_buffer[const], unsigned long num_sents_in_buffer, unsigned long vocab_size, char **unique_words, struct_map **ngram_map, struct_map_word_class **word2class_map);
struct_sent_info parse_input_line(char * restrict line_in, const struct_sent_info sent_info_a, struct_map **ngram_map);
float query_sents_in_buffer(const struct cmd_args cmd_args, char * restrict sent_buffer[const], const unsigned long num_sents_in_buffer, struct_map **ngram_map, struct_map_word_class **word2class_map);

#endif // INCLUDE_HEADER

#ifndef INCLUDE_CLUSTERCAT_HEADER
#define INCLUDE_CLUSTERCAT_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
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
#define UNKNOWN_WORD_ID      0
#define UNKNOWN_WORD         "UnK"
// Number of characters to read-in for each line
#define BUFLEN 8192
#define STDIN_SENT_MAX_CHARS 40000
#define STDIN_SENT_MAX_WORDS 1024
#define MAX_WORD_LEN 255

typedef unsigned short sentlen_t; // Number of words in a sentence
//typedef unsigned short wclass_t;  // Defined in clustercat-map.h
//typedef unsigned int   word_id_t; // Defined in clustercat-map.h
#define SENT_LEN_MAX USHRT_MAX

enum class_algos {EXCHANGE, BROWN};

#include "clustercat-data.h" // bad. chicken-and-egg typedef deps

typedef struct {
	char **sent;
	wclass_t class_sent[STDIN_SENT_MAX_WORDS];
	unsigned int sent_counts[STDIN_SENT_MAX_WORDS];
	short word_lengths[STDIN_SENT_MAX_WORDS];
	sentlen_t length;
} struct_sent_info;

typedef struct {
	word_id_t * restrict sent;
	wclass_t * restrict class_sent;
	unsigned int * restrict sent_counts;
	sentlen_t length;
} struct_sent_int_info;

typedef struct {
	unsigned long token_count;
	unsigned long line_count;
	word_id_t     type_count;
} struct_model_metadata;

char *argv_0_basename; // Allow for global access to filename

struct cmd_args {
	unsigned long  max_tune_sents;
	char*          dev_file;
	wclass_t       num_classes;
	unsigned short tune_cycles : 10;
	unsigned char  class_order : 6;
	unsigned short num_threads : 10;
	unsigned short min_count : 9;
	char           verbose : 3;     // Negative values increasingly suppress normal output
	unsigned char  class_algo : 2;  // enum class_algos
};

void sent_store_string2sent_store_int(struct_map_word **ngram_map, char * restrict sent_store_string[restrict], struct_sent_int_info sent_store_int[restrict], const unsigned long num_sents_in_store);
void populate_word_ids(struct_map_word **ngram_map, char * restrict unique_words[const], const word_id_t type_count);
void build_word_count_array(struct_map_word **ngram_map, char * restrict unique_words[const], unsigned int word_counts[restrict], const word_id_t type_count);

void increment_ngram_variable_width(struct_map_word **ngram_map, char * restrict sent[const], const short * restrict word_lengths, short start_position, const sentlen_t i);
void increment_ngram_fixed_width(struct_map_class **map, wclass_t class_sent[const], short start_position, const sentlen_t i);
unsigned long copy_buffer_to_store(char * restrict sent_buffer[const], const unsigned long num_sents_in_buffer, char * restrict * restrict sent_store, unsigned long num_sents_in_store, const unsigned long max_tune_sents);
void process_int_sents_in_store(const struct_sent_int_info * const sent_store_int, const unsigned long num_sents_in_buffer, const wclass_t word2class[const], struct_map_class **class_map, const word_id_t temp_word, const wclass_t temp_class);
unsigned long process_str_sents_in_buffer(char * restrict sent_buffer[], const unsigned long num_sents_in_buffer);
unsigned long process_str_sent(char * restrict sent_str);
word_id_t filter_infrequent_words(const struct cmd_args cmd_args, struct_model_metadata * restrict model_metadata, struct_map_word ** ngram_map);
void tokenize_sent(char * restrict sent_str, struct_sent_info *sent_info);
void init_clusters(const struct cmd_args cmd_args, word_id_t vocab_size, wclass_t word2class[restrict]);
void cluster(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, wclass_t word2class[]);
struct_sent_info parse_input_line(char * restrict line_in, const char * restrict temp_word, const wclass_t temp_class);
double query_int_sents_in_store(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const wclass_t word2class[const], struct_map_class **class_map, const word_id_t temp_word, const wclass_t temp_class);

void print_sent_info(struct_sent_info * restrict sent_info);
#endif // INCLUDE_HEADER

#ifndef INCLUDE_CLUSTERCAT_HEADER
#define INCLUDE_CLUSTERCAT_HEADER

#include <math.h>

void increment_ngram(struct_map **ngram_map, char * restrict sent[const], const short * restrict word_lengths, short start_position, const sentlen_t i);
unsigned long process_sents_in_buffer(char * restrict sent_buffer[], const long num_sents_in_buffer);
unsigned long process_sent(char * restrict sent_str);
void tokenize_sent(char * restrict sent_str, struct_sent_info *sent_info);
static inline void free_sent_info_local(struct_sent_info sent_info);

enum class_algos {EXCHANGE, BROWN};
struct cmd_args {
	long           max_sents_in_buffer;
	unsigned short num_threads : 10;
	char           verbose : 3;    // Negative values increasingly suppress normal output
	unsigned char  class_algo : 2; // enum class_algos
};

#endif // INCLUDE_HEADER

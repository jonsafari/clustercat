#ifndef INCLUDE_CC_CLUSTER_HEADER
#define INCLUDE_CC_CLUSTER_HEADER

#include "clustercat.h"

void cluster(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const unsigned int word_counts[const], char * word_list[restrict], wclass_t word2class[], struct_word_bigram_entry * restrict word_bigrams, struct_word_bigram_entry * restrict word_bigrams_rev, unsigned int * restrict word_class_counts, unsigned int * restrict word_class_rev_counts);

#endif // INCLUDE_HEADER

#ifndef INCLUDE_CC_DBG_HEADER
#define INCLUDE_CC_DBG_HEADER

#include "clustercat.h"

void print_word_class_counts(const struct cmd_args cmd_args, const struct_model_metadata model_metadata, const unsigned int * restrict word_class_counts);

void print_word_bigrams(const struct cmd_args cmd_args, const struct_model_metadata model_metadata, const struct_word_bigram_entry * restrict word_bigrams);

#endif // INCLUDE_HEADER

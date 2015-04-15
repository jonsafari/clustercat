#ifndef INCLUDE_CLUSTERCAT_IO
#define INCLUDE_CLUSTERCAT_IO

#include "clustercat.h"
#include "clustercat-data.h"

// Import
long fill_sent_buffer(FILE *file, char * restrict sent_buffer[], const long max_sents_in_buffer, size_t * memusage);
struct_model_metadata process_input(FILE *file, struct_map_word ** initial_word_map, struct_map_bigram ** initial_bigram_map, size_t *memusage);

#endif // INCLUDE_HEADER

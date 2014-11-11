#ifndef INCLUDE_CLUSTERCAT_IO
#define INCLUDE_CLUSTERCAT_IO

#include "clustercat.h"
#include "clustercat-data.h"

// Import
struct_model_metadata dklm_import_model_file(char * restrict file_name, struct_map **word_map, struct_map **word_word_map, DATA_STRUCT_FLOAT_TYPE DATA_STRUCT_FLOAT_NAME, struct_map **ngram_map, struct_map **class_map);
long fill_sent_buffer(FILE *file, char * restrict sent_buffer[], const long max_sents_in_buffer);

#endif // INCLUDE_HEADER

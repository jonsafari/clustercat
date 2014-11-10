#ifndef INCLUDE_DKLM_IO
#define INCLUDE_DKLM_IO

#include "dklm.h"
#include "dklm-data.h"

#define WEIGHTS_SUFFIX ".weights"
#define WEIGHTS_FILE_NAME_LENGTH 1000

// Import
struct_model_metadata dklm_import_model_file(char * restrict file_name, struct_map **word_map, struct_map **word_word_map, DATA_STRUCT_FLOAT_TYPE DATA_STRUCT_FLOAT_NAME, struct_map **ngram_map, struct_map **class_map);
void dklm_import_class_file(const char * restrict file_name, struct_map_word_class **word2class_map);
long fill_sent_buffer(FILE *file, char * restrict sent_buffer[], const long max_sents_in_buffer);
void import_weights(const double full_weights_array[const], const struct_model_metadata model_metadata, struct_weights *weights, struct_dklm_params *dklm_params);
unsigned int import_weights_from_file(const struct_model_metadata model_metadata, double full_weights_array[]);

// Export
void export_weights_to_file(char * restrict model_file_name, const double full_weights_array[const], const unsigned int length);

#endif // INCLUDE_HEADER

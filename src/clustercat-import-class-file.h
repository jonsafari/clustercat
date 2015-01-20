#ifndef INCLUDE_CLUSTERCAT_IMPORT_CLASS_FILE_HEADER
#define INCLUDE_CLUSTERCAT_IMPORT_CLASS_FILE_HEADER

#include "clustercat.h" // wclass_t

void import_class_file(const struct cmd_args cmd_args, word_id_t vocab_size, wclass_t word2class[restrict], const char * restrict class_file_name);

#endif // INCLUDE_HEADER

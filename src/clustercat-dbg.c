#include "clustercat-dbg.h"

void print_word_class_counts(const struct cmd_args cmd_args, const struct_model_metadata model_metadata, const unsigned int * restrict word_class_counts) {
	for (wclass_t class = 0; class < cmd_args.num_classes; class++) {
		printf("Class=%u:\n\t", class);
		for (word_id_t word = 0; word < model_metadata.type_count; word++) {
			printf("#(<%u,%hu>)=%u  ", word, class, (word * cmd_args.num_classes + class));
		}
		printf("\n");
	}
	fflush(stdout);
}

void print_word_bigrams(const struct cmd_args cmd_args, const struct_model_metadata model_metadata, const struct_word_bigram_entry * restrict word_bigrams) {
	;
}

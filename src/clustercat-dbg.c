#include "clustercat-dbg.h"

void print_word_class_counts(const struct cmd_args cmd_args, const struct_model_metadata model_metadata, const word_class_count_t * restrict word_class_counts) {
	for (wclass_t class = 0; class < cmd_args.num_classes; class++) {
		printf("Class=%u   Offsets=%u,%u,...%u:\n\t", class, class, class+cmd_args.num_classes, (model_metadata.type_count-1) * cmd_args.num_classes + class);
		for (word_id_t word = 0; word < model_metadata.type_count; word++) {
			printf("#(<%u,%hu>)=%u  ", word, class, word_class_counts[word * cmd_args.num_classes + class]);
		}
		printf("\n");
	}
	fflush(stdout);
}

void print_word_bigrams(const struct cmd_args cmd_args, const struct_model_metadata model_metadata, const struct_word_bigram_entry * restrict word_bigrams) {
	;
}

#include <stdio.h>
#include <errno.h>
#include "clustercat-import-class-file.h"

// Parse TSV file input and overwrite relevant word mappings
void import_class_file(const struct cmd_args cmd_args, word_id_t vocab_size, wclass_t word2class[restrict], const char * restrict class_file_name) {
	char * restrict line_end;
	char * restrict line = calloc(MAX_WORD_LEN + 9, 1);

	FILE *file = fopen(class_file_name, "r");
	if (!file) {
		fprintf(stderr, "%s: fopen of '%s' failed: %s.\n", argv_0_basename, class_file_name, strerror(errno));
		exit(EXIT_FAILURE);
	}
#if 0
	while (gzgets(file, line, BUFLEN - 1) != 0) {

		line_end = strchr(line, '\n');
		*line_end = '\0';

		// Parse each line
		unsigned int keylen = strcspn(line, PRIMARY_SEP_STRING);
		line[keylen] = '\0'; // Split key and count
		char * restrict key = line;
		char * restrict class = line + keylen + 1;
		//printf("keylen=%i, key=<<%s>>, class=<<%d>>\n", keylen, key, class);
		map_add_class(word2class_map, key, class);
	}

	// Add start/end of sentence <s> and </s>
	map_update_class(word2class_map, "<s>", "<s>");
	map_update_class(word2class_map, "</s>", "</s>");

#endif
	fclose(file);
	free(line);
}

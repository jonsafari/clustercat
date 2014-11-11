#include <zlib.h>		// Strongly recommended to use zlib-1.2.5 or newer
#include <stdio.h>
#include "dklm.h"
#include "dklm-data.h"
#include "dklm-array.h"
#include "dklm-io.h"

/// Import
// Parse file input and build hash map
struct_model_metadata dklm_import_model_file(char * restrict file_name, struct_map **word_map, struct_map **word_word_map, DATA_STRUCT_FLOAT_TYPE DATA_STRUCT_FLOAT_NAME, struct_map **ngram_map, struct_map **class_map) {
	char * restrict line_end;
	char * restrict line = (char *) calloc((uInt) BUFLEN, 1);
	struct_model_metadata model_metadata = {0};
	model_metadata.file_name = file_name;

	gzFile file = gzopen(file_name, "rb");
	if (!file) {
		fprintf(stderr, "%s: gzopen of '%s' failed: %s.\n", argv_0_basename, file_name, strerror(errno));
		exit(EXIT_FAILURE);
	}

	while (gzgets(file, line, BUFLEN - 1) != 0) {

		line_end = strchr(line, '\n');
		*line_end = '\0';

		unsigned int keylen = strcspn(line, PRIMARY_SEP_STRING);
		line[keylen] = '\0'; // Split key and count
		char * restrict key = line;
		//char key[keylen+1];
		//strncpy(key, line, keylen);
		//key[keylen] = NULL;
		unsigned int count = (unsigned int) atol(line + keylen + 1);
		//char * restrict count = line + keylen+1;
		//printf("keylen=%i, key=<<%s>>, count=<<%d>>\n", keylen, key, count);
		if (strncmp(key, "#NG ", 5) == 0) { // Process n-gram entries
			key = key + 5;
			map_update_entry(ngram_map, key, count);
		} else if (strncmp(key, "#CL ", 8) == 0) { // Process n-gram entries
			key = key + 8;
			map_update_entry(class_map, key, count);
		} else if (strncmp(key, "#ToKeNs", 10) == 0) { // Metadata
			model_metadata.token_count += count;
		} else if (strncmp(key, "#LiNeS", 9) == 0) { // Metadata
			model_metadata.line_count += count;
		} else if (strncmp(key, "#Ngram_Order", 15) == 0) { // Metadata
			model_metadata.ngram_order = count;
		} else if (strncmp(key, "#Class_Order", 15) == 0) { // Metadata
			model_metadata.class_order = count;
		} else if (strncmp(key, "#DK_Order", 13) == 0) { // Metadata
			model_metadata.dklm_order = count;
		} else if (strncmp(key, "#DK_Probs", 13) == 0) { // Metadata
			model_metadata.dklm_probs = count;
		} else if (strncmp(key, "#DK_02-gram_Hist", 23) == 0) { // Metadata
			model_metadata.dklm_hist_2 = count;
		} else if (strncmp(key, "#DK_03-gram_Hist", 23) == 0) { // Metadata
			model_metadata.dklm_hist_3 = count;
		} else if (strncmp(key, "#", 2) == 0) { // Other metadata
		} else if (keylen == strcspn(key, SECONDARY_SEP_STRING)) { // Unigrams (no spaces in key)
			map_update_entry(word_map, key, count);

		} else { // Word-word pairs
			if (model_metadata.dklm_probs) { // DKLM probabilities
				float frac_count = (float) atof(line + keylen + 1);
				UPDATE_ENTRY_FLOAT(DATA_STRUCT_FLOAT_NAME, key, frac_count);
			}
			else // DKLM counts
				map_update_entry(word_word_map, key, count);
		}
		//printf(" key=<<%s>>, count=%i\n", key, count);
	}
	gzclose(file);
	free(line);
	model_metadata.type_count = map_count(word_map);
	return model_metadata;
}

long fill_sent_buffer(FILE *file, char * restrict sent_buffer[], const long max_sents_in_buffer) {
	char line_in[STDIN_SENT_MAX_CHARS];
	long sent_buffer_num = 0;
	unsigned int strlen_line_in = 0;

	while (sent_buffer_num < max_sents_in_buffer) {
		//line_in = readline(""); // Use GNU Readline; slow
		//if (! line_in)
		//	break;
		if (! fgets(line_in, STDIN_SENT_MAX_CHARS, file))
			break;
		else {
			strlen_line_in = strlen(line_in); // We'll need this a couple times
			if (strlen_line_in == STDIN_SENT_MAX_CHARS-1)
				fprintf(stderr, "\n%s: Warning: Input line too long, at buffer line %li. The full line was:\n%s\n", argv_0_basename, sent_buffer_num+1, line_in);
			sent_buffer[sent_buffer_num] = (char *)malloc(1+ strlen_line_in * sizeof(char *));
			strncpy(sent_buffer[sent_buffer_num], line_in, 1+strlen_line_in);
			//printf("fill_sent_buffer 7: sent_buffer_num=%li, line_in=<<%s>>, strlen_line_in=%u, sent_in=<<%s>>\n", sent_buffer_num, line_in, strlen_line_in, sent_buffer[sent_buffer_num]); fflush(stdout);
			sent_buffer_num++;
		}
	}
	//printf("fill_sent_buffer 99: sent_buffer_num=%li\n", sent_buffer_num);
	return sent_buffer_num;
}


void export_weights_to_file(char * restrict model_file_name, const double full_weights_array[const], const unsigned int length) {
	char file_name[WEIGHTS_FILE_NAME_LENGTH]; sprintf(file_name, "%s%s", model_file_name, WEIGHTS_SUFFIX);
	FILE *file = fopen(file_name, "w");
	fprint_array(file, full_weights_array, length, " ");
	fclose(file);
}

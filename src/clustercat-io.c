#include <stdio.h>
#include "clustercat.h"
#include "clustercat-data.h"
#include "clustercat-array.h"
#include "clustercat-io.h"

struct_model_metadata process_input(FILE *file, struct_map_word ** initial_word_map, struct_map_word ** initial_bigram_map, size_t *memuage) {
	struct_model_metadata model_metadata = {0};
	char curr_word[MAX_WORD_LEN + 1]; curr_word[MAX_WORD_LEN] = '\0';
	char bigram[2*MAX_WORD_LEN + 2]; strncpy(bigram, "<s>", 4); bigram[2*MAX_WORD_LEN+1] = '\0';
	int ch, prev_ch = 0;
	unsigned int curr_word_pos = 0;
	unsigned int prev_word_pos = 0;

	while (!feof(file)) {
		ch = getc(file);
		if (ch == ' ' || ch == '\t') { // in between words
			if (prev_ch == ' ' || prev_ch == 0) { // ignore multiple spaces or leading spaces
				prev_ch = ' ';
				continue;
			} else {
				// increment current word in word map
				curr_word[curr_word_pos] = '\0'; // terminate word
				map_increment_count(initial_word_map, curr_word); // increment </s> in word map
				// increment previous+current bigram in bigram map

				model_metadata.token_count++;
				strncpy(bigram, curr_word, MAX_WORD_LEN + 1);
				printf("curr_word=<<%s>>; bigram=<<%s>>\n", curr_word, bigram); fflush(stdout);
				//curr_word[0] = '\0'; // truncate word
				curr_word_pos = 0;
			}
		} else if (ch == '\n') { // end of line
			//if (prev_ch == ' ' || prev_ch == 0) { // ignore trailing spaces or leading spaces
			//	prev_ch = 0;
			//} else {
				prev_ch = 0;
				model_metadata.line_count++;
				map_increment_count(initial_word_map, "</s>"); // increment </s> in word map
				//strncat(bigram, " </s>", 6);
				//map_increment_count(&initial_bigram_map, bigram); // increment previous+</s> bigram in bigram map
			//}
			curr_word_pos = 0;
			strncpy(bigram, "<s>", 4);
		} else { // normal character;  within a word
			if (curr_word_pos > MAX_WORD_LEN) { // word is too long; do nothing until space or newline
				continue;
			} else {
				curr_word[curr_word_pos++] = ch;
				prev_ch = ch;
			}
		}
	}

	printf("line_count=%lu; token_count=%lu; type_count=%u\n", model_metadata.line_count, model_metadata.token_count, model_metadata.type_count);
	return model_metadata;
}

long fill_sent_buffer(FILE *file, char * restrict sent_buffer[], const long max_sents_in_buffer, size_t * memusage) {
	char line_in[STDIN_SENT_MAX_CHARS];
	long sent_buffer_num = 0;
	unsigned int strlen_line_in = 0;

	while (sent_buffer_num < max_sents_in_buffer) {
		if (! fgets(line_in, STDIN_SENT_MAX_CHARS, file))
			break;
		else {
			strlen_line_in = strlen(line_in); // We'll need this a couple times;  strnlen isn't in C standard :-(
			if (strlen_line_in == STDIN_SENT_MAX_CHARS-1)
				fprintf(stderr, "\n%s: Notice: Input line too long (> %lu chars), at buffer line %li. The line started with:    %.250s ...\n", argv_0_basename, (long unsigned int) STDIN_SENT_MAX_CHARS, sent_buffer_num+1, line_in);
			sent_buffer[sent_buffer_num] = (char *)malloc(1+ strlen_line_in * sizeof(char *));
			*memusage += 1+ strlen_line_in * sizeof(char *);
			strncpy(sent_buffer[sent_buffer_num], line_in, 1+strlen_line_in);
			//printf("fill_sent_buffer 7: sent_buffer_num=%li, line_in=<<%s>>, strlen_line_in=%u, sent_in=<<%s>>\n", sent_buffer_num, line_in, strlen_line_in, sent_buffer[sent_buffer_num]); fflush(stdout);
			sent_buffer_num++;
		}
	}
	//printf("fill_sent_buffer 99: sent_buffer_num=%li\n", sent_buffer_num);
	return sent_buffer_num;
}

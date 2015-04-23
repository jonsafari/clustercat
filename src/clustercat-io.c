#include <stdio.h>
#include "clustercat.h"
#include "clustercat-data.h"
#include "clustercat-array.h"
#include "clustercat-io.h"

struct_model_metadata process_input(FILE *file, struct_map_word ** initial_word_map, struct_map_bigram ** initial_bigram_map, size_t *memusage) {
	struct_model_metadata model_metadata = {0};
	char curr_word[MAX_WORD_LEN + 1]; curr_word[MAX_WORD_LEN] = '\0';
	register unsigned int chars_in_sent = 0;
	register int ch, prev_ch = 0;
	unsigned int curr_word_pos = 0;
	map_update_count(initial_word_map, UNKNOWN_WORD, 0, 0); // initialize entry for <unk>, <s>, and </s>
	map_update_count(initial_word_map, "<s>", 0, 1);
	map_update_count(initial_word_map, "</s>", 0, 2);
	const word_id_t start_id = map_find_id(initial_word_map, "<s>", 1);
	const word_id_t end_id = map_find_id(initial_word_map, "</s>", 2);
	const size_t sizeof_struct_map_word   = sizeof(struct_map_word);
	const size_t sizeof_struct_map_bigram = sizeof(struct_map_bigram);
	unsigned int prev_word_id = start_id;
	model_metadata.type_count = 3; // start with <unk>, <s>, and </s>, and <unk>.

	while (!feof(file)) {
		ch = getc(file);
		//printf("«%c» ", ch); fflush(stdout);
		if (ch == ' ' || ch == '\t' || ch == '\n') { // end of a word

			curr_word[curr_word_pos] = '\0'; // terminate word
			if (!strncmp(curr_word, "", 1)) { // ignore empty words, due to leading, trailing, and multiple spaces
				//printf("skipping empty word; ch=«%c»\n", ch); fflush(stdout);
				if (ch == '\n') { // trailing spaces require more stuff to do
					const struct_word_bigram bigram = {prev_word_id, end_id};
					if (map_increment_bigram(initial_bigram_map, &bigram)) // increment previous+</s> bigram in bigram map
						*memusage += sizeof_struct_map_bigram;
					prev_ch = 0;
					chars_in_sent = 0;
					prev_word_id = start_id;
					model_metadata.line_count++;
				}
				continue;
			}
			//printf("curr_word=%s, prev_id=%u\n", curr_word, prev_word_id); fflush(stdout);
			model_metadata.token_count++;
			curr_word_pos = 0;
			// increment current word in word map
			const word_id_t curr_word_id = map_increment_count(initial_word_map, curr_word, model_metadata.type_count); // <unk>'s word_id is set to 0.

			if (curr_word_id == model_metadata.type_count) { // previous call to map_increment_count() had a new word
				model_metadata.type_count++;
				*memusage += sizeof_struct_map_word;
			}

			// increment previous+current bigram in bigram map
			const struct_word_bigram bigram = {prev_word_id, curr_word_id};
			//printf("{%u,%u}\n", prev_word_id, curr_word_id); fflush(stdout);
			if (map_increment_bigram(initial_bigram_map, &bigram)) // true if bigram is new
				*memusage += sizeof_struct_map_bigram;

			//printf("process_input(): curr_word=<<%s>>; curr_word_id=%u, prev_word_id=%u\n", curr_word, curr_word_id, prev_word_id); fflush(stdout);
			if (ch == '\n') { // end of line
				const struct_word_bigram bigram = {curr_word_id, end_id};
				if (map_increment_bigram(initial_bigram_map, &bigram)) // increment previous+</s> bigram in bigram map
					*memusage += sizeof_struct_map_bigram;
				prev_ch = 0;
				chars_in_sent = 0;
				prev_word_id = start_id;
				model_metadata.line_count++;
			} else {
				prev_word_id = curr_word_id;
			}

		} else { // normal character;  within a word
			if (curr_word_pos > MAX_WORD_LEN) { // word is too long; do nothing until space or newline
				continue;
			} else {
				curr_word[curr_word_pos++] = ch;
				prev_ch = ch;
			}
		}
	}

	// Set counts of <s> and </s> once, based on line_count
	map_update_count(initial_word_map, "<s>", model_metadata.line_count, 1);
	map_update_count(initial_word_map, "</s>", model_metadata.line_count, 2);
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

#include <zlib.h>		// Strongly recommended to use zlib-1.2.5 or newer
#include <stdio.h>
#include "clustercat.h"
#include "clustercat-data.h"
#include "clustercat-array.h"
#include "clustercat-io.h"

long fill_sent_buffer(FILE *file, char * restrict sent_buffer[], const long max_sents_in_buffer) {
	char line_in[STDIN_SENT_MAX_CHARS];
	long sent_buffer_num = 0;
	unsigned int strlen_line_in = 0;

	while (sent_buffer_num < max_sents_in_buffer) {
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

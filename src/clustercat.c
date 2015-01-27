/** Induces word categories
 *  By Jon Dehdari, 2014
 *  Usage: ./clustercat [options] < corpus.tok.txt > classes.tsv
**/

#include <limits.h>				// UCHAR_MAX, UINT_MAX
#include <float.h>				// DBL_MAX, etc.
#include <math.h>				// isnan()
#include <time.h>				// clock_t, clock(), CLOCKS_PER_SEC
#include <stdbool.h>
#include <locale.h>				// OPTIONAL!  Comment-out on non-Posix machines, and the function setlocale() in the first line of main()

#include "clustercat.h"						// Model importing/exporting functions
#include "clustercat-array.h"				// which_maxf()
#include "clustercat-data.h"
#include "clustercat-import-class-file.h"	// import_class_file()
#include "clustercat-io.h"					// fill_sent_buffer()
#include "clustercat-math.h"				// perplexity(), powi()
#include "clustercat-ngram-prob.h"			// class_ngram_prob()

#define USAGE_LEN 10000

// Declarations
void get_usage_string(char * restrict usage_string, int usage_len);
void parse_cmd_args(const int argc, char **argv, char * restrict usage, struct cmd_args *cmd_args);
void free_sent_info(struct_sent_info sent_info);
char * restrict class_algo         = NULL;
char * restrict initial_class_file = NULL;
char * restrict weights_string     = NULL;

struct_map_word *ngram_map = NULL; // Must initialize to NULL
struct_map_word_class *word2class_map = NULL; // Must initialize to NULL;  This can be global since we only update it after finding best exchange.  We can use a local conditional for thread-specific class counting.
char usage[USAGE_LEN];
size_t memusage = 0;


// Defaults
struct cmd_args cmd_args = {
	.class_algo             = EXCHANGE,
	.dev_file               = NULL,
	.max_tune_sents         = 1000000,
	.min_count              = 2,
	.max_array              = 3,
	.class_order            = 3,
	.num_threads            = 4,
	.num_classes            = 0,
	.tune_cycles            = 15,
	.verbose                = 0,
};



int main(int argc, char **argv) {
	setlocale(LC_ALL, ""); // Comment-out on non-Posix systems
	clock_t time_start = clock();
	time_t time_t_start;
	time(&time_t_start);
	argv_0_basename = basename(argv[0]);
	weights_string = "0.3 0.175 0.05 0.175 0.3";
	get_usage_string(usage, USAGE_LEN); // This is a big scary string, so build it elsewhere

	//printf("sizeof(cmd_args)=%zd\n", sizeof(cmd_args));
	parse_cmd_args(argc, argv, usage, &cmd_args);

	struct_model_metadata global_metadata;
	global_metadata.token_count = 0;
	global_metadata.line_count  = 0;


	// The list of unique words should always include <s>, unknown word, and </s>
	map_update_count(&ngram_map, UNKNOWN_WORD, 0); // Should always be first
	map_update_count(&ngram_map, "<s>", 0);
	map_update_count(&ngram_map, "</s>", 0);

	char * * restrict sent_buffer = calloc(sizeof(char **), cmd_args.max_tune_sents);
	if (sent_buffer == NULL) {
		fprintf(stderr,  "%s: Error: Unable to allocate enough memory for initial sentence buffer.  %'lu MB needed.  Reduce --tune-sents (current value: %lu)\n", argv_0_basename, ((sizeof(void *) * cmd_args.max_tune_sents) / 1048576 ), cmd_args.max_tune_sents); fflush(stderr);
		exit(7);
	}
	memusage += sizeof(void *) * cmd_args.max_tune_sents;

	unsigned long num_sents_in_buffer = 0; // We might need this number later if a separate dev set isn't provided;  we'll just tune on final buffer.

	while (1) {
		// Fill sentence buffer
		num_sents_in_buffer = fill_sent_buffer(stdin, sent_buffer, cmd_args.max_tune_sents);
		//printf("cmd_args.max_tune_sents=%lu; global_metadata.line_count=%lu; num_sents_in_buffer=%lu\n", cmd_args.max_tune_sents, global_metadata.line_count, num_sents_in_buffer);
		if ((num_sents_in_buffer == 0) || ( cmd_args.max_tune_sents <= global_metadata.line_count)) // No more sentences in buffer
			break;

		global_metadata.line_count  += num_sents_in_buffer;
		global_metadata.token_count += process_str_sents_in_buffer(sent_buffer, num_sents_in_buffer);
	}

	global_metadata.type_count        = map_count(&ngram_map);

	// Filter out infrequent words
	word_id_t number_of_deleted_words = filter_infrequent_words(cmd_args, &global_metadata, &ngram_map);

	// Check or set number of classes
	if (cmd_args.num_classes >= global_metadata.type_count) { // User manually set number of classes is too low
		fprintf(stderr, "%s: Error: Number of classes (%u) is not less than vocabulary size (%u).  Decrease the value of --num-classes\n", argv_0_basename, cmd_args.num_classes, global_metadata.type_count); fflush(stderr);
		exit(3);
	} else if (cmd_args.num_classes == 0) { // User did not manually set number of classes at all
		cmd_args.num_classes = (wclass_t) sqrt(global_metadata.type_count);
	}

	// Get list of unique words
	char * * restrict word_list = (char **)malloc(sizeof(char*) * global_metadata.type_count);
	memusage += sizeof(char*) * global_metadata.type_count;
	get_keys(&ngram_map, word_list);

	// Build array of word_counts
	unsigned int * restrict word_counts = malloc(sizeof(unsigned int) * global_metadata.type_count);
	memusage += sizeof(unsigned int) * global_metadata.type_count;
	build_word_count_array(&ngram_map, word_list, word_counts, global_metadata.type_count);

	// Now that we have filtered-out infrequent words, we can populate values of struct_map_word->word_id values.  We could have merged this step with get_keys(), but for code clarity, we separate it out.  It's a one-time, quick operation.
	populate_word_ids(&ngram_map, word_list, global_metadata.type_count);

	struct_sent_int_info * restrict sent_store_int = malloc(sizeof(struct_sent_int_info) * global_metadata.line_count);
	if (sent_store_int == NULL) {
		fprintf(stderr,  "%s: Error: Unable to allocate enough memory for sent_store_int.  Reduce --tune-sents (current value: %lu)\n", argv_0_basename, cmd_args.max_tune_sents); fflush(stderr);
		exit(8);
	}
	memusage += sizeof(struct_sent_int_info) * global_metadata.line_count;
	sent_buffer2sent_store_int(&ngram_map, sent_buffer, sent_store_int, global_metadata.line_count);
	// Each sentence in sent_buffer was freed within sent_buffer2sent_store_int().  Now we can free the entire array
	free(sent_buffer);
	memusage -= sizeof(void *) * cmd_args.max_tune_sents;

	// Initialize and set word bigram counts
	clock_t time_bigram_start = clock();
	if (cmd_args.verbose >= 0)
		fprintf(stderr, "%s: Bigram counting ... ", argv_0_basename); fflush(stderr);
	struct_word_bigram ** restrict word_bigrams = calloc(sizeof(void *), global_metadata.type_count);
	memusage += sizeof(void *) * global_metadata.type_count;
	size_t bigram_memusage = set_bigram_counts(cmd_args, word_bigrams, sent_store_int, global_metadata.line_count);
	memusage += bigram_memusage;
	clock_t time_bigram_end = clock();
	if (cmd_args.verbose >= 0)
		fprintf(stderr, "in %'.2f secs.  Bigram memusage: %zu B (sizeof(struct_word_bigram)=%zu x %g unique bigrams)\n", (double)(time_bigram_end - time_bigram_start)/CLOCKS_PER_SEC, bigram_memusage, sizeof(struct_word_bigram), bigram_memusage / (float)sizeof(struct_word_bigram)); fflush(stderr);

	// Initialize clusters, and possibly read-in external class file
	wclass_t * restrict word2class = malloc(sizeof(wclass_t) * global_metadata.type_count);
	memusage += sizeof(wclass_t) * global_metadata.type_count;
	init_clusters(cmd_args, global_metadata.type_count, word2class);
	if (initial_class_file != NULL)
		import_class_file(&ngram_map, global_metadata.type_count, word2class, initial_class_file, cmd_args.num_classes); // Overwrite subset of word mappings, from user-provided initial_class_file
	delete_all(&ngram_map);

	// Calculate memusage for count_arrays
	for (unsigned char i = 1; i <= cmd_args.max_array; i++) {
		memusage += cmd_args.num_threads * (powi(cmd_args.num_classes, i) * sizeof(unsigned int));
	}

	clock_t time_model_built = clock();
	if (cmd_args.verbose >= -1)
		fprintf(stderr, "%s: Finished loading %'lu tokens and %'u types (%'u filtered) from %'lu lines in %'.2f secs\n", argv_0_basename, global_metadata.token_count, global_metadata.type_count, number_of_deleted_words, global_metadata.line_count, (double)(time_model_built - time_start)/CLOCKS_PER_SEC); fflush(stderr);
	if (cmd_args.verbose >= -1)
		fprintf(stderr, "%s: Approximate mem usage: %'.1fMB\n", argv_0_basename, (double)memusage / 1048576); fflush(stderr);

	cluster(cmd_args, sent_store_int, global_metadata, word_counts, word_list, word2class);

	// Now print the final word2class_map
	if (cmd_args.verbose >= 0)
		print_words_and_classes(global_metadata.type_count, word_list, word2class);

	clock_t time_clustered = clock();
	time_t time_t_end;
	time(&time_t_end);
	double time_secs_total = difftime(time_t_end, time_t_start);
	if (cmd_args.verbose >= -1)
		fprintf(stderr, "%s: Finished clustering in %'.2f CPU seconds.  Total time about %.0fm %is\n", argv_0_basename, (double)(time_clustered - time_model_built)/CLOCKS_PER_SEC, time_secs_total/60, ((int)time_secs_total % 60)  );

	free(word2class);
	free(word_bigrams);
	free(word_list);
	free(word_counts);
	free(sent_store_int);
	exit(0);
}


void get_usage_string(char * restrict usage_string, int usage_len) {

	snprintf(usage_string, usage_len, "ClusterCat  (c) 2014-2015 Jon Dehdari - LGPL v3 or Mozilla Public License v2\n\
\n\
Usage:    clustercat [options] < corpus.tok.txt > classes.tsv \n\
\n\
Function: Induces word categories from plaintext\n\
\n\
Options:\n\
     --class-algo <s>     Set class-induction algorithm {brown,exchange} (default: exchange)\n\
     --class-file <file>  Initialize exchange word classes from a tsv file (default: pseudo-random initialization for exchange)\n\
     --dev-file <file>    Use separate file to tune on (default: training set, from stdin)\n\
 -h, --help               Print this usage\n\
 -j, --jobs <hu>          Set number of threads to run simultaneously (default: %d threads)\n\
     --min-count <hu>     Minimum count of entries in training set to consider (default: %d occurrences)\n\
     --max-array <c>      Set maximum order of n-grams for which to use an array instead of a sparse hash map (default: %d-grams)\n\
 -n, --num-classes <c>    Set number of word classes (default: square root of vocabulary size)\n\
 -q, --quiet              Print less output.  Use additional -q for even less output\n\
     --tune-sents <lu>    Set size of sentence store to tune on (default: first %'lu sentences)\n\
     --tune-cycles <hu>   Set max number of cycles to tune on (default: %d cycles)\n\
 -v, --verbose            Print additional info to stderr.  Use additional -v for more verbosity\n\
 -w, --weights 'f f ...'  Set class interpolation weights for: 3-gram, 2-gram, 1-gram, rev 2-gram, rev 3-gram. (default: %s)\n\
\n\
", cmd_args.num_threads, cmd_args.min_count, cmd_args.max_array, cmd_args.max_tune_sents, cmd_args.tune_cycles, weights_string);
}
// -o, --order <i>          Maximum n-gram order in training set to consider (default: %d-grams)\n\

void parse_cmd_args(int argc, char **argv, char * restrict usage, struct cmd_args *cmd_args) {
	for (int arg_i = 1; arg_i < argc; arg_i++) {
		if (!(strcmp(argv[arg_i], "-h") && strcmp(argv[arg_i], "--help"))) {
			printf("%s", usage);
			exit(0);
		} else if (!strcmp(argv[arg_i], "--class-algo")) {
			char * restrict class_algo_string = argv[arg_i+1];
			arg_i++;
			if (!strcmp(class_algo_string, "brown"))
				cmd_args->class_algo = BROWN;
			else if (!strcmp(class_algo_string, "exchange"))
				cmd_args->class_algo = EXCHANGE;
			else { printf("%s", usage); exit(0); }
		} else if (!strcmp(argv[arg_i], "--class-file")) {
			initial_class_file = argv[arg_i+1];
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--dev-file")) {
			cmd_args->dev_file = argv[arg_i+1];
			printf("Bug Jon to implement --dev-file!\n"); fflush(stderr);
			exit(1);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-j") && strcmp(argv[arg_i], "--jobs"))) {
			cmd_args->num_threads = (unsigned int) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--min-count")) {
			cmd_args->min_count = (unsigned int) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--max-array")) {
			cmd_args->max_array = (unsigned char) atol(argv[arg_i+1]);
			if ((cmd_args->max_array) < 1 || (cmd_args->max_array > 3)) {
				printf("%s: --max-array value should be between 1-3\n", argv_0_basename);
				fflush(stderr);
				exit(10);
			}
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-n") && strcmp(argv[arg_i], "--num-classes"))) {
			cmd_args->num_classes = (wclass_t) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-o") && strcmp(argv[arg_i], "--order"))) {
			cmd_args->class_order = (unsigned char) atoi(argv[arg_i+1]);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-q") && strcmp(argv[arg_i], "--quiet"))) {
			cmd_args->verbose--;
		} else if (!strcmp(argv[arg_i], "--tune-sents")) {
			cmd_args->max_tune_sents = atol(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--tune-cycles")) {
			cmd_args->tune_cycles = (unsigned short) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-v") && strcmp(argv[arg_i], "--verbose"))) {
			cmd_args->verbose++;
		} else if (!(strcmp(argv[arg_i], "-w") && strcmp(argv[arg_i], "--weights"))) {
			weights_string = argv[arg_i+1];
			arg_i++;
		} else if (!strncmp(argv[arg_i], "-", 1)) { // Unknown flag
			printf("%s: Unknown command-line argument: %s\n\n", argv_0_basename, argv[arg_i]);
			printf("%s", usage); fflush(stderr);
			exit(2);
		}
	}
}

void sent_buffer2sent_store_int(struct_map_word **ngram_map, char * restrict sent_buffer[restrict], struct_sent_int_info sent_store_int[restrict], const unsigned long num_sents_in_store) {
	for (unsigned long i = 0; i < num_sents_in_store; i++) { // Copy string-oriented sent_buffer[] to int-oriented sent_store_int[]
		if (sent_buffer[i] == NULL) // No more sentences in buffer
			break;

		char * restrict sent_i = sent_buffer[i];
		//printf("sent[%lu]=<<%s>>\n", i, sent_buffer[i]); fflush(stdout);

		word_id_t sent_int_temp[SENT_LEN_MAX];

		// Stupid strtok is destructive
		char * restrict pch = NULL;
		pch = strtok(sent_i, TOK_CHARS);

		// Initialize first element in sentence to <s>
		sent_int_temp[0]        = map_find_int(ngram_map, "<s>");

		sentlen_t w_i = 1; // Word 0 is <s>; we initialize it here to be able to use it after the loop for </s>

		for (; pch != NULL  &&  w_i < SENT_LEN_MAX; w_i++) {
			if (w_i == STDIN_SENT_MAX_WORDS - 1) { // Deal with pathologically-long lines
				//fprintf(stderr, "%s: Warning: Line %lu length at %u. Truncating pathologically-long line starting with: \"%s ...\"\n", argv_0_basename, i+1, w_i, sent_i);
				break;
			}

			sent_int_temp[w_i] = map_find_int(ngram_map, pch);
			//printf("pch=%s, int=%u, count=%u\n", pch, sent_int_temp[w_i], sent_counts_int_temp[w_i]);

			pch = strtok(NULL, TOK_CHARS);
		}

		// Initialize first element in sentence to </s>
		sent_int_temp[w_i]        = map_find_int(ngram_map, "</s>");

		sentlen_t sent_length = w_i + 1; // Include <s>;  we use this local variable for perspicuity later on
		sent_store_int[i].length = sent_length;

		// Now that we know the actual sentence length, we can allocate the right amount for the sentence
		sent_store_int[i].sent = malloc(sizeof(word_id_t) * sent_length);

		memusage += sizeof(word_id_t) * sent_length;
		memusage += sizeof(wclass_t) * sent_length;
		memusage += sizeof(unsigned int) * sent_length;

		// Copy the temporary fixed-width array on stack to dynamic-width array in heap
		memcpy(sent_store_int[i].sent, sent_int_temp, sizeof(word_id_t) * sent_length);

		free(sent_i); // Free-up string-based sentence
	}
}

void build_word_count_array(struct_map_word **ngram_map, char * restrict word_list[const], unsigned int word_counts[restrict], const word_id_t type_count) {
	for (word_id_t i = 0; i < type_count; i++) {
		word_counts[i] = map_find_count(ngram_map, word_list[i]);
	}
}

void populate_word_ids(struct_map_word **ngram_map, char * restrict word_list[const], const word_id_t type_count) {
	for (word_id_t i = 0; i < type_count; i++) {
		//printf("%s=%u\n", word_list[i], i);
		map_set_word_id(ngram_map, word_list[i], i);
	}
}

word_id_t filter_infrequent_words(const struct cmd_args cmd_args, struct_model_metadata * restrict model_metadata, struct_map_word ** ngram_map) {

	unsigned long number_of_deleted_words = 0;
	unsigned long vocab_size = model_metadata->type_count; // Save this to separate variable since we'll modify model_metadata.type_count later

	// Get keys
	// Iterate over keys
	//   If count of key_i < threshold,
	//     increment count of <unk> by count of key_i,
	//     decrement model_metadata.type_count by one
	//     free & delete entry in map,

	char **local_word_list = (char **)malloc(model_metadata->type_count * sizeof(char*));
	//char * local_word_list[model_metadata->type_count];
	if (vocab_size != get_keys(ngram_map, local_word_list)) {
		printf("Error: model_metadata->type_count != get_keys()\n"); fflush(stderr);
		exit(4);
	}

	for (unsigned long word_i = 0; word_i < vocab_size; word_i++) {
		unsigned long word_i_count = map_find_count(ngram_map, local_word_list[word_i]);  // We'll use this a couple times
		if ((word_i_count < cmd_args.min_count) && (strncmp(local_word_list[word_i], UNKNOWN_WORD, MAX_WORD_LEN)) ) { // Don't delete <unk>
			number_of_deleted_words++;
			map_update_count(ngram_map, UNKNOWN_WORD, word_i_count);
			if (cmd_args.verbose > 2)
				printf("Filtering-out word: %s (%lu < %hu);\tcount(%s)=%u\n", local_word_list[word_i], word_i_count, cmd_args.min_count, UNKNOWN_WORD, map_find_count(ngram_map, UNKNOWN_WORD));
			model_metadata->type_count--;
			struct_map_word *local_s;
			HASH_FIND_STR(*ngram_map, local_word_list[word_i], local_s);
			delete_entry(ngram_map, local_s);
		}
		//else
			//printf("Keeping word: %s (%lu < %hu);\tcount(%s)=%u\n", local_word_list[word_i], word_i_count, cmd_args.min_count, UNKNOWN_WORD, map_find_count(ngram_map, UNKNOWN_WORD));
	}

	free(local_word_list);
	return number_of_deleted_words;
}

void increment_ngram_variable_width(struct_map_word **ngram_map, char * restrict sent[const], const short * restrict word_lengths, short start_position, const sentlen_t i) {
	short j;
	size_t sizeof_char = sizeof(char); // We use this multiple times
	unsigned char ngram_len = 0; // Terminating char '\0' is same size as joining tab, so we'll just count that later

	// We first build the longest n-gram string, then successively remove the leftmost word

	for (j = i; j >= start_position ; j--) { // Determine length of longest n-gram string, starting with smallest to ensure longest string is less than 255 chars
		if (ngram_len + sizeof_char + word_lengths[j]  < UCHAR_MAX ) { // Ensure n-gram string is less than 255 chars
			ngram_len += sizeof_char + word_lengths[j]; // the additional sizeof_char is for either a space for words in the history, or for a \0 for word_i
		} else { // It's too big; ensmallen n-gram
			start_position++;
		}

		//printf("increment_ngram1: start_position=%d, ngram_len=%u, j=%d, len(j)=%u, w_j=%s, i=%i, w_i=%s\n", start_position, ngram_len, j, word_lengths[j], sent[j], i, sent[i]);
	}

	if (!ngram_len) // We couldn't do anything with this n-gram because it was too long.  Wa, wa, wa
		return;

	char ngram[ngram_len];
	strcpy(ngram, sent[start_position]);
	// For single words, append \0 instead of space
	if (start_position < i)
		strcat(ngram, SECONDARY_SEP_STRING);
	else
		ngram[ngram_len] = '\0';
	//printf("increment_ngram1.5: start_position=%d, i=%i, w_i=%s, ngram_len=%d, ngram=<<%s>>\n", start_position, i, sent[i], ngram_len, ngram);

	for (j = start_position+1; j <= i ; ++j) { // Build longest n-gram string.  We do this as a separate loop than before since malloc'ing a bunch of times is probably more expensive than the first cheap loop
		strcat(ngram, sent[j]);
		if (j < i) // But wait! There's more!
			strcat(ngram, SECONDARY_SEP_STRING);
	}
	//printf("increment_ngram3: start_position=%d, i=%i, w_i=%s, ngram_len=%d, ngram=<<%s>>\n", start_position, i, sent[i], ngram_len, ngram);

	char * restrict jp = ngram;
	short diff = i - start_position;
	for (j = start_position; j <= i; ++j, --diff) { // Traverse longest n-gram string
		//if (cmd_args.verbose)
			//printf("increment_ngram4: start_position=%d, i=%i, w_i=%s, ngram_len=%d, ngram=<<%s>>, jp=<<%s>>\n", start_position, i, sent[i], ngram_len, ngram, jp);
		map_increment_count(ngram_map, jp);
		//if (diff > 0) // 0 allows for unigrams
			jp += sizeof_char + word_lengths[j];
	}
}

void increment_ngram_fixed_width(const struct cmd_args cmd_args, count_arrays_t count_arrays, wclass_t sent[const], short start_position, const sentlen_t i) {

	// n-grams handled using a dense array for each n-gram order
	for (unsigned char ngram_len = i - start_position + 1; ngram_len > 0; ngram_len--) { // Unigrams in count_arrays[0], ...
	//printf(" incr._ngram_fw5: sent[i-1]=%u, sent[i]=%u, ngram_len=%u, [%hu,%hu,%hu], offset=%zu\n", sent[i-1], sent[i], ngram_len, sent[i+1-ngram_len], sent[i+2-ngram_len], sent[i+3-ngram_len], array_offset(&sent[i+1-ngram_len], ngram_len, cmd_args.num_classes)); fflush(stdout);
		//array_offset(&sent[i+1-ngram_len], ngram_len, cmd_args.num_classes);
		count_arrays[ngram_len-1][ array_offset(&sent[i+1-ngram_len], ngram_len, cmd_args.num_classes) ]++;
		//printf(" incr._ngram_fw6: arr: start_pos=%d, i=%i, w_i=%u, ngram_len=%d, class_ngram[0]=%hu, new count=%u\n", start_position, i, sent[i], ngram_len, sent[i], count_arrays[ngram_len-1][ array_offset(&sent[i+1-ngram_len], ngram_len, cmd_args.num_classes) ] );
	}
}

void tally_int_sents_in_store(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const wclass_t word2class[const], count_arrays_t count_arrays, struct_map_class **class_map, const word_id_t temp_word, const wclass_t temp_class) {

	for (unsigned long current_sent_num = 0; current_sent_num < model_metadata.line_count; current_sent_num++) { // loop over sentences
		register sentlen_t sent_length = sent_store_int[current_sent_num].length;
		register word_id_t word_id;
		wclass_t class_sent[STDIN_SENT_MAX_WORDS];

		for (sentlen_t i = 0; i < sent_length; i++) { // loop over words
			word_id = sent_store_int[current_sent_num].sent[i];
			if (word_id == temp_word) { // This word matches the temp word
				class_sent[i] = temp_class;
			} else { // This word doesn't match temp word
				class_sent[i] = word2class[word_id];
			}

			sentlen_t start_position_class = (i >= CLASSLEN-1) ? i - (CLASSLEN-1) : 0; // N-grams starting point is 0, for <s>
			increment_ngram_fixed_width(cmd_args, count_arrays, class_sent, start_position_class, i);
		}
	}
}

unsigned long process_str_sents_in_buffer(char * restrict sent_buffer[], const unsigned long num_sents_in_buffer) {
	unsigned long token_count = 0;
	char local_sent_copy[STDIN_SENT_MAX_CHARS];
	local_sent_copy[STDIN_SENT_MAX_CHARS-1] = '\0'; // Ensure at least last element of array is terminating character

	for (unsigned long current_sent_num = 0; current_sent_num < num_sents_in_buffer; current_sent_num++) {
		strncpy(local_sent_copy, sent_buffer[current_sent_num], STDIN_SENT_MAX_CHARS-2); // Strtok, which is used later, is destructive
		token_count += process_str_sent(local_sent_copy);
	}

	return token_count;
}

unsigned long process_str_sent(char * restrict sent_str) { // Uses global ngram_map
	if (!strncmp(sent_str, "\n", 1)) // Ignore empty lines
		return 0;

	struct_sent_info sent_info = {0};
	sent_info.sent = malloc(STDIN_SENT_MAX_WORDS * sizeof(char*));

	// We could have built up the word n-gram counts directly from sent_str, but it's
	// the only one out of the three models we're building that we can do this way, and
	// it's simpler to have a more uniform way of building these up.

	tokenize_sent(sent_str, &sent_info);
	unsigned long token_count = sent_info.length;
	//if (cmd_args.verbose > 2) {
	//	print_sent_info(&sent_info);
	//	fflush(stdout);
	//}

	register sentlen_t i;
	for (i = 0; i < sent_info.length; i++) {
		increment_ngram_variable_width(&ngram_map, sent_info.sent, sent_info.word_lengths, i, i); // N-grams starting point is 0, for <s>;  We only need unigrams for visible words
	}

	free(sent_info.sent);
	return token_count;
}


void tokenize_sent(char * restrict sent_str, struct_sent_info *sent_info) {
	// Stupid strtok is destructive
	char * restrict pch = NULL;
	pch = strtok(sent_str, TOK_CHARS);

	// Initialize first element in sentence to <s>
	sent_info->sent[0] = "<s>";
	sent_info->word_lengths[0]  = strlen("<s>");

	sentlen_t w_i = 1; // Word 0 is <s>

	for (; pch != NULL  &&  w_i < SENT_LEN_MAX; w_i++) {
		if (w_i == STDIN_SENT_MAX_WORDS - 1) { // Deal with pathologically-long lines
			fprintf(stderr, "%s: Warning: Truncating pathologically-long line starting with: \"%s %s %s %s %s %s ...\"\n", argv_0_basename, sent_info->sent[1], sent_info->sent[2], sent_info->sent[3], sent_info->sent[4], sent_info->sent[5], sent_info->sent[6]);
			break;
		}

		sent_info->sent[w_i] = pch;
		sent_info->word_lengths[w_i] = strlen(pch);
		//printf("pch=%s; len=%u\n", pch, sent_info->word_lengths[w_i]);

		if (sent_info->word_lengths[w_i] > MAX_WORD_LEN) { // Deal with pathologically-long words
			pch[MAX_WORD_LEN] = '\0';
			sent_info->word_lengths[w_i] = MAX_WORD_LEN;
			fprintf(stderr, "%s: Warning: Truncating pathologically-long word '%s'\n", argv_0_basename, pch);
		}

		pch = strtok(NULL, TOK_CHARS);
	}

	// Initialize last element in sentence to </s>
	sent_info->sent[w_i] = "</s>";
	sent_info->word_lengths[w_i]  = strlen("</s>");
	sent_info->length = w_i + 1; // Include <s>
}

// Slightly different from free_sent_info() since we don't free the individual words in sent_info.sent here
void free_sent_info(struct_sent_info sent_info) {
	for (sentlen_t i = 1; i < sent_info.length-1; ++i) // Assumes word_0 is <s> and word_sentlen is </s>, which weren't malloc'd
		free(sent_info.sent[i]);

	free(sent_info.sent);
}

size_t set_bigram_counts(const struct cmd_args cmd_args, struct_word_bigram ** restrict word_bigrams, const struct_sent_int_info * const sent_store_int, const unsigned long line_count) {
	register size_t memusage = 0;
	register size_t sizeof_struct_word_bigram = sizeof(struct_word_bigram);

	for (unsigned long current_sent_num = 0; current_sent_num < line_count; current_sent_num++) { // loop over sentences
		register sentlen_t sent_length = sent_store_int[current_sent_num].length;
		register word_id_t word_id_i;
		register word_id_t word_id_i_minus_1;

		for (sentlen_t i = 1; i < sent_length; i++) { // loop over words in a sentence, starting with the first word after <s>
			word_id_i         = sent_store_int[current_sent_num].sent[i];
			word_id_i_minus_1 = sent_store_int[current_sent_num].sent[i-1];
			if (word_bigrams[word_id_i_minus_1] == 0) { // word_i-1 doesn't have any successors yet
				word_bigrams[word_id_i_minus_1] = calloc(sizeof_struct_word_bigram,1);
				word_bigrams[word_id_i_minus_1]->word_id = word_id_i;
				memusage += sizeof_struct_word_bigram;
			} else { // Check to see if we've seen this bigram before
				struct_word_bigram * bigram = word_bigrams[word_id_i_minus_1];
				while (bigram->word_id != word_id_i) { // We've seen this bigram before.  Stop
					if (bigram->next == NULL) { // No more existing bigrams; add new one
						bigram->next = calloc(sizeof_struct_word_bigram,1);
						(bigram->next)->word_id = word_id_i;
						memusage += sizeof_struct_word_bigram;
						break; // in loop we'd break here
					} else { // We have reached the end of the linked list yet; try the next one
						bigram = bigram->next;
					}
				}
			}
		}
	}

	return memusage;
}

void init_clusters(const struct cmd_args cmd_args, word_id_t vocab_size, wclass_t word2class[restrict]) {
	register unsigned long word_i = 0;

	if (cmd_args.class_algo == EXCHANGE) { // It doesn't really matter how you initialize word classes in exchange algo.  This assigns words from the word list an incrementing class number from [0,num_classes].  So it's a simple pseudo-randomized initialization.
		register wclass_t class = 1; // 0 is reserved
		for (; word_i < vocab_size; word_i++, class++) {
			if (class > cmd_args.num_classes)
				class = 1;
			//printf("cls=%u, w_i=%lu, vocab_size=%u\n", class, word_i, vocab_size);
			word2class[word_i] = class;
		}

	} else if (cmd_args.class_algo == BROWN) { // Really simple initialization: one class per word
		for (unsigned long class = 1; word_i < vocab_size; word_i++, class++)
			word2class[word_i] = class;
	}
}

void cluster(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const unsigned int word_counts[const], char * word_list[restrict], wclass_t word2class[]) {
	unsigned long steps = 0;

	if (cmd_args.class_algo == EXCHANGE) { // Exchange algorithm: See Sven Martin, Jörg Liermann, Hermann Ney. 1998. Algorithms For Bigram And Trigram Word Clustering. Speech Communication 24. 19-37. http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.2354
		// Get initial logprob
		struct_map_class *class_map = NULL; // Build local counts of classes, for flexibility
		count_arrays_t count_arrays = malloc(cmd_args.max_array * sizeof(void *));
		init_count_arrays(cmd_args, count_arrays);
		tally_int_sents_in_store(cmd_args, sent_store_int, model_metadata, word2class, count_arrays, &class_map, -1, 0); // Get class ngram counts. We use -1 so that no words are ever substituted
		//printf("42: "); for (wclass_t i = 0; i <= cmd_args.num_classes; i++) {
		//	printf("c_%u=%u, ", i, count_arrays[0][i]);
		//} printf("\n"); fflush(stdout);
		double best_log_prob = query_int_sents_in_store(cmd_args, sent_store_int, model_metadata, word_counts, word2class, word_list, count_arrays, &class_map, -1, 1);
		free_count_arrays(cmd_args, count_arrays);
		free(count_arrays);

		if (cmd_args.verbose >= -1)
			fprintf(stderr, "%s: Expected Steps:  %'lu (%'u word types x %'u classes x %'u cycles);  initial logprob=%g, PP=%g\n", argv_0_basename, (unsigned long)model_metadata.type_count * cmd_args.num_classes * cmd_args.tune_cycles, model_metadata.type_count, cmd_args.num_classes, cmd_args.tune_cycles, best_log_prob, perplexity(best_log_prob, (model_metadata.token_count - model_metadata.line_count))); fflush(stderr);

		unsigned short cycle = 1; // Keep this around afterwards to print out number of actually-completed cycles
		for (; cycle <= cmd_args.tune_cycles; cycle++) {
			bool end_cycle_short = true; // This gets set to false if any word's class changes

			if (cmd_args.verbose >= -1)
				fprintf(stderr, "%s: Starting cycle %u with logprob=%g, PP=%g\n", argv_0_basename, cycle, best_log_prob, perplexity(best_log_prob,(model_metadata.token_count - model_metadata.line_count))); fflush(stderr);
			//#pragma omp parallel for num_threads(cmd_args.num_threads) reduction(+:steps) // non-determinism
			for (word_id_t word_i = 0; word_i < model_metadata.type_count; word_i++) {
				//wclass_t unknown_word_class  = get_class(&word2class_map, UNKNOWN_WORD, UNKNOWN_WORD_CLASS); // We'll use this later
				//wclass_t unknown_word_class  = word2class[UNKNOWN_WORD_ID]; // We'll use this later
				const wclass_t old_class = word2class[word_i];
				double log_probs[cmd_args.num_classes]; // This doesn't need to be private in the OMP parallelization since each thead is writing to different element in the array

				#pragma omp parallel for num_threads(cmd_args.num_threads) reduction(+:steps)
				for (wclass_t class = 1; class <= cmd_args.num_classes; class++) { // class values range from 1 to cmd_args.num_classes, so we need to add/subtract by one in various places below when dealing with arrays
					steps++;
					// Get log prob
					struct_map_class *class_map = NULL; // Build local counts of classes, for flexibility
					count_arrays_t count_arrays = malloc(cmd_args.max_array * sizeof(void *));  // This array is small and dense
					init_count_arrays(cmd_args, count_arrays);
					tally_int_sents_in_store(cmd_args, sent_store_int, model_metadata, word2class, count_arrays, &class_map, word_i, class); // Get class ngram counts
					//log_probs[class-1] = query_sents_in_store(cmd_args, sent_store, model_metadata, &class_map, word, class);
					log_probs[class-1] = query_int_sents_in_store(cmd_args, sent_store_int, model_metadata, word_counts, word2class, word_list, count_arrays, &class_map, word_i, class);
					delete_all_class(&class_map); // Individual elements in map are malloc'd, so we need to free all of them
					free_count_arrays(cmd_args, count_arrays);
					free(count_arrays);
				}

				const wclass_t best_hypothesis_class = 1 + which_max(log_probs, cmd_args.num_classes);
				const double best_hypothesis_log_prob = max(log_probs, cmd_args.num_classes);

				if (cmd_args.verbose > 1) {
					printf("Orig logprob for word w_«%u» using class «%hu» is %g;  Hypos %u-%u: ", word_i, old_class, log_probs[old_class-1], 1, cmd_args.num_classes);
					fprint_array(stdout, log_probs, cmd_args.num_classes, ","); fflush(stdout);
					if (best_hypothesis_log_prob > 0) { // Shouldn't happen
						fprintf(stderr, "Error: best_hypothesis_log_prob=%g for class %hu > 0\n", best_hypothesis_log_prob, best_hypothesis_class); fflush(stderr);
						exit(9);
					}
				}

				if (log_probs[old_class-1] < best_hypothesis_log_prob) { // We've improved
					end_cycle_short = false;

					if (cmd_args.verbose > 0)
						fprintf(stderr, " Moving id=%-6u %-18s %u -> %u\t(logprob %g -> %g)\n", word_i, word_list[word_i], old_class, best_hypothesis_class, log_probs[old_class-1], best_hypothesis_log_prob); fflush(stderr);
					//word2class[word_i] = best_hypothesis_class;
					//map_update_class(&word2class_map, word, best_hypothesis_class);
					word2class[word_i] = best_hypothesis_class;
					best_log_prob = best_hypothesis_log_prob;
				}
			}

			// In principle if there's no improvement in the determinitistic exchange algo, we can stop cycling; there will be no more gains
			if (end_cycle_short)
				break;
		}
		if (cmd_args.verbose >= -1)
			fprintf(stderr, "%s: Completed steps: %'lu (%'u word types x %'u classes x %'u cycles);     best logprob=%g, PP=%g\n", argv_0_basename, steps, model_metadata.type_count, cmd_args.num_classes, cycle-1, best_log_prob, perplexity(best_log_prob,(model_metadata.token_count - model_metadata.line_count))); fflush(stderr);

	} else if (cmd_args.class_algo == BROWN) { // Agglomerative clustering.  Stops when the number of current clusters is equal to the desired number in cmd_args.num_classes
		// "Things equal to nothing else are equal to each other." --Anon
		for (unsigned long current_num_classes = model_metadata.type_count; current_num_classes > cmd_args.num_classes; current_num_classes--) {
			for (word_id_t word_i = 0; word_i < model_metadata.type_count; word_i++) {
				float log_probs[cmd_args.num_classes];
				//#pragma omp parallel for num_threads(cmd_args.num_threads)
				for (wclass_t class = 0; class < cmd_args.num_classes; class++, steps++) {
					// Get log prob
					log_probs[class] = -1 * (class+1); // Dummy predicate
				}
				wclass_t best_class = which_maxf(log_probs, cmd_args.num_classes);
				printf("Moving w_%u to class %u\n", word_i, best_class);
			}
		}
	}
}



double query_int_sents_in_store(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const unsigned int word_counts[const], const wclass_t word2class[const], char * word_list[restrict], const count_arrays_t count_arrays, struct_map_class **class_map, const word_id_t temp_word, const wclass_t temp_class) {
	double sum_log_probs = 0.0; // For perplexity calculation

	unsigned long current_sent_num;
	//#pragma omp parallel for private(current_sent_num) num_threads(cmd_args.num_threads) reduction(+:sum_log_probs)
	for (current_sent_num = 0; current_sent_num < model_metadata.line_count; current_sent_num++) {

		register sentlen_t sent_length = sent_store_int[current_sent_num].length;
		register word_id_t word_id;
		wclass_t class_sent[STDIN_SENT_MAX_WORDS];

		// Build array of classes
		for (sentlen_t i = 0; i < sent_length; i++) { // loop over words
			word_id = sent_store_int[current_sent_num].sent[i];
			if (word_id == temp_word) { // This word matches the temp word
				class_sent[i] = temp_class;
			} else { // This word doesn't match temp word
				class_sent[i] = word2class[word_id];
			}
		}

		float sent_score = 0.0; // Initialize with identity element

		const struct_sent_int_info * const sent_info = &sent_store_int[current_sent_num];


		for (sentlen_t i = 1; i < sent_length; i++) {
			const word_id_t word_i = sent_info->sent[i];
			const wclass_t class_i = class_sent[i];
			//wclass_t class_i_entry[CLASSLEN] = {0};
			//class_i_entry[0] = class_i;
			const unsigned int word_i_count = word_counts[word_i];
			//const unsigned int class_i_count = map_find_count_fixed_width(class_map, class_i_entry);
			const unsigned int class_i_count = count_arrays[0][class_i-1];
			//float word_i_count_for_next_freq_score = word_i_count ? word_i_count : 0.2; // Using a very small value for unknown words messes up distribution
			if (cmd_args.verbose > 1) {
				printf("qry_snts_n_stor: i=%d\tcnt=%d\tcls=%u\tcls_cnt=%d\tw_id=%u\tw=%s\n", i, word_i_count, class_i, class_i_count, word_i, word_list[word_i]);
				fflush(stdout);
				if (class_i_count < word_i_count) { // Shouldn't happen
					printf("Error: class_%hu_count=%u < word_id[%u]_count=%u\n", class_i, class_i_count, word_i, word_i_count); fflush(stderr);
					exit(5);
				}
			}

			// Class prob is transition prob * emission prob
			const float emission_prob = word_i_count ? (float)word_i_count / (float)class_i_count :  1 / (float)class_i_count;


			// Calculate transition probs
			float weights_class[] = {0.35, 0.14, 0.02, 0.14, 0.35};
			//float weights_class[] = {0.3, 0.175, 0.05, 0.0, 0.0};
			//float weights_class[] = {0.0, 0.95, 0.05, 0.0, 0.0};
			float order_probs[5] = {0};
			order_probs[2] = class_i_count / (float)model_metadata.token_count; // unigram probs
			float sum_weights = weights_class[2]; // unigram prob will always occur
			float sum_probs = weights_class[2] * order_probs[2]; // unigram prob will always occur

			//const float transition_prob = class_ngram_prob(cmd_args, count_arrays, class_map, i, class_i, class_i_count, class_sent, CLASSLEN, model_metadata, weights_class);
			if (i > 1) { // Need at least "<s> w_1" in history
				order_probs[0] = count_arrays[2][ array_offset(&class_sent[i-2], 3, cmd_args.num_classes) ] / (float)count_arrays[1][ array_offset(&class_sent[i-1], 2, cmd_args.num_classes) ]; // trigram probs
				order_probs[0] = isnan(order_probs[0]) ? 0.0f : order_probs[0]; // If the bigram history is 0, result will be a -nan
				sum_weights += weights_class[0];
				sum_probs += weights_class[0] * order_probs[0];
			}

			// We'll always have at least "<s>" in history
			order_probs[1] = count_arrays[1][ array_offset(&class_sent[i-1], 2, cmd_args.num_classes) ] / (float)count_arrays[0][ array_offset(&class_sent[i], 1, cmd_args.num_classes) ]; // bigram probs
			//printf("order_probs[1] = %u / %u\n", count_arrays[1][ array_offset(&class_sent[i], 2, cmd_args.num_classes) ], count_arrays[0][ array_offset(&class_sent[i], 1, cmd_args.num_classes)]);
			sum_weights += weights_class[1];
			sum_probs += weights_class[1] * order_probs[1];

			if (i < sent_length-1) { // Need at least "</s>" to the right
				order_probs[3] = count_arrays[1][ array_offset(&class_sent[i], 2, cmd_args.num_classes) ] / (float)count_arrays[0][ array_offset(&class_sent[i+1], 1, cmd_args.num_classes) ]; // future bigram probs
				sum_weights += weights_class[3];
				sum_probs += weights_class[3] * order_probs[3];
			}

			if (i < sent_length-2) { // Need at least "w </s>" to the right
			order_probs[4] = count_arrays[2][ array_offset(&class_sent[i], 3, cmd_args.num_classes) ] / (float)count_arrays[1][ array_offset(&class_sent[i+1], 2, cmd_args.num_classes) ]; // future trigram probs
			order_probs[4] = isnan(order_probs[4]) ? 0.0f : order_probs[4]; // If the bigram history is 0, result will be a -nan
				sum_weights += weights_class[4];
				sum_probs += weights_class[4] * order_probs[4];
			}
			const float transition_prob = sum_probs / sum_weights;
			const float class_prob = emission_prob * transition_prob;


			if (cmd_args.verbose > 1) {
				printf(" w_id=%u, w_i_cnt=%g, class_i=%u, class_i_count=%i, emission_prob=%g, transition_prob=%g, class_prob=%g, log2=%g, sum_probs=%g, sum_weights=%g\n", word_i, (float)word_i_count, class_i, class_i_count, emission_prob, transition_prob, class_prob, log2f(class_prob), sum_probs, sum_weights);
				printf("transition_probs:\t");
				fprint_arrayf(stdout, order_probs, 5, ","); fflush(stdout);
				if (class_i_count > model_metadata.token_count) { // Shouldn't happen
					printf("Error: prob of order max_ngram_used > 1;  %u/%lu\n", class_i_count, model_metadata.token_count); fflush(stderr);
					exit(6);
				}
				if (! ((class_prob >= 0) && (class_prob <= 1))) {
					printf("Error: prob is not within [0,1]  %g\n", class_prob); fflush(stderr);
					exit(11);
				}
			}

			sent_score += log2((double)class_prob); // Increment running sentence total;  we can use doubles for global-level scores

		} // for i loop

		sum_log_probs += sent_score; // Increment running test set total, for perplexity
	} // Done querying current sentence
	return sum_log_probs;
}

void print_sent_info(struct_sent_info * restrict sent_info) {
	printf("struct sent_info { length = %u\n", sent_info->length);
	for (sentlen_t i = 0; i < sent_info->length; i++) {
		printf(" i=%u\twlen=%i\tw=%s\n", i, sent_info->word_lengths[i], sent_info->sent[i]);
	}
	printf("}\n");
}

void init_count_arrays(const struct cmd_args cmd_args, count_arrays_t count_arrays) {
	for (unsigned char i = 1; i <= cmd_args.max_array; i++) { // Start with unigrams in count_arrays[0], ...
		count_arrays[i-1] = calloc(powi(cmd_args.num_classes, i), sizeof(unsigned int)); // powi() is in clustercat-math.c
		if (count_arrays[i-1] == NULL) {
			fprintf(stderr,  "%s: Error: Unable to allocate enough memory for %u-grams.  I tried to allocate %zu MB per thread (%zuB * %u^%u).  Reduce the number of desired classes using --num-classes (current value: %u)\n", argv_0_basename, i, sizeof(unsigned int) * powi(cmd_args.num_classes, i) / 1048576, sizeof(unsigned int), cmd_args.num_classes, i, cmd_args.num_classes ); fflush(stderr);
			exit(12);
		}
		//printf("Allocating %zu B (cmd_args.num_classes=%u^i=%u * sizeof(uint)=%zu)\n", (powi(cmd_args.num_classes, i) * sizeof(unsigned int)), cmd_args.num_classes, i, sizeof(unsigned int));
	}
}

void free_count_arrays(const struct cmd_args cmd_args, count_arrays_t count_arrays) {
	for (unsigned char i = 1; i <= cmd_args.max_array; i++) { // Start with unigrams in count_arrays[0], ...
		free(count_arrays[i-1]);
	}
}

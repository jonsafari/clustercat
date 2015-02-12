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
#include "clustercat-dbg.h"					// for printing out various complex data structures
#include "clustercat-import-class-file.h"	// import_class_file()
#include "clustercat-io.h"					// fill_sent_buffer()
#include "clustercat-math.h"				// perplexity(), powi()
#include "clustercat-ngram-prob.h"			// class_ngram_prob()

#define USAGE_LEN 10000

// Declarations
void get_usage_string(char * restrict usage_string, int usage_len);
void parse_cmd_args(const int argc, char **argv, char * restrict usage, struct cmd_args *cmd_args);
void free_sent_info(struct_sent_info sent_info);
char * restrict class_algo           = NULL;
char * restrict in_train_file_string = NULL;
char * restrict out_file_string      = NULL;
char * restrict initial_class_file   = NULL;
char * restrict weights_string       = NULL;

struct_map_word *ngram_map = NULL; // Must initialize to NULL
char usage[USAGE_LEN];
size_t memusage = 0;


// Defaults
struct cmd_args cmd_args = {
	.class_algo        = EXCHANGE,
	.class_offset      = 0,
	.max_tune_sents    = 10000000,
	.min_count         = 2,
	.max_array         = 3,
	.num_threads       = 4,
	.num_classes       = 0,
	.rev_alternate     = 3,
	.tune_cycles       = 15,
	.unidirectional    = false,
	.verbose           = 0,
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

	// Fill sentence buffer
	FILE *in_train_file = stdin;
	if (in_train_file_string)
		in_train_file = fopen(in_train_file_string, "r");
	const unsigned long num_sents_in_buffer = fill_sent_buffer(in_train_file, sent_buffer, cmd_args.max_tune_sents);
	fclose(in_train_file);
	//printf("cmd_args.max_tune_sents=%lu; global_metadata.line_count=%lu; num_sents_in_buffer=%lu\n", cmd_args.max_tune_sents, global_metadata.line_count, num_sents_in_buffer);
	global_metadata.line_count  += num_sents_in_buffer;
	if (cmd_args.max_tune_sents <= global_metadata.line_count) { // There are more sentences in stdin than were processed
		fprintf(stderr, "%s: Warning: Sentence buffer is full.  You probably should increase it using --tune-sents .  Current value: %lu\n", argv_0_basename, cmd_args.max_tune_sents); fflush(stderr);
	}

	global_metadata.token_count += process_str_sents_in_buffer(sent_buffer, num_sents_in_buffer);
	global_metadata.type_count   = map_count(&ngram_map);

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
	sort_by_count(&ngram_map); // Speeds up lots of stuff later
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
	memusage += sent_buffer2sent_store_int(&ngram_map, sent_buffer, sent_store_int, global_metadata.line_count);
	// Each sentence in sent_buffer was freed within sent_buffer2sent_store_int().  Now we can free the entire array
	free(sent_buffer);
	memusage -= sizeof(void *) * cmd_args.max_tune_sents;


	// Initialize clusters, and possibly read-in external class file
	wclass_t * restrict word2class = malloc(sizeof(wclass_t) * global_metadata.type_count);
	memusage += sizeof(wclass_t) * global_metadata.type_count;
	init_clusters(cmd_args, global_metadata.type_count, word2class, word_counts, word_list);
	if (initial_class_file != NULL)
		import_class_file(&ngram_map, global_metadata.type_count, word2class, initial_class_file, cmd_args.num_classes); // Overwrite subset of word mappings, from user-provided initial_class_file
	delete_all(&ngram_map);


	// Initialize and set word bigram listing
	clock_t time_bigram_start = clock();
	size_t bigram_memusage = 0; size_t bigram_rev_memusage = 0;
	struct_word_bigram_entry * restrict word_bigrams = NULL;
	struct_word_bigram_entry * restrict word_bigrams_rev = NULL;
	if (cmd_args.verbose >= -1)
		fprintf(stderr, "%s: Word bigram listing ... ", argv_0_basename); fflush(stderr);

	#pragma omp parallel sections // Both bigram listing and reverse bigram listing can be done in parallel
	{
		#pragma omp section
		{
			word_bigrams = calloc(global_metadata.type_count, sizeof(struct_word_bigram_entry));
			memusage += sizeof(struct_word_bigram_entry) * global_metadata.type_count;
			bigram_memusage = set_bigram_counts(cmd_args, word_bigrams, sent_store_int, global_metadata.line_count, false);
		}

		// Initialize and set *reverse* word bigram listing
		#pragma omp section
		{
			if (cmd_args.rev_alternate) { // Don't bother building this if it won't be used
				word_bigrams_rev = calloc(global_metadata.type_count, sizeof(struct_word_bigram_entry));
				memusage += sizeof(struct_word_bigram_entry) * global_metadata.type_count;
				bigram_rev_memusage = set_bigram_counts(cmd_args, word_bigrams_rev, sent_store_int, global_metadata.line_count, true);
			}
		}
	}

	memusage += bigram_memusage + bigram_rev_memusage;
	clock_t time_bigram_end = clock();
	if (cmd_args.verbose >= -1)
		fprintf(stderr, "in %'.2f CPU secs.  Bigram memusage: %'.1f MB\n", (double)(time_bigram_end - time_bigram_start)/CLOCKS_PER_SEC, (bigram_memusage + bigram_rev_memusage)/(double)1048576); fflush(stderr);


	// Build <v,c> counts, which consists of a word followed by a given class
	unsigned int * restrict word_class_counts = calloc(1 + cmd_args.num_classes * global_metadata.type_count , sizeof(unsigned int));
	if (word_class_counts == NULL) {
		fprintf(stderr,  "%s: Error: Unable to allocate enough memory for <v,c>.  %'.1f MB needed.  Maybe increase --min-count\n", argv_0_basename, ((cmd_args.num_classes * global_metadata.type_count * sizeof(unsigned int)) / (double)1048576 )); fflush(stderr);
		exit(13);
	}
	memusage += cmd_args.num_classes * global_metadata.type_count * sizeof(unsigned int);
	fprintf(stderr, "%s: Allocating %'.1f MB for word_class_counts: num_classes=%u x type_count=%u x sizeof(uint)=%zu\n", argv_0_basename, (double)(cmd_args.num_classes * global_metadata.type_count * sizeof(unsigned int)) / 1048576 , cmd_args.num_classes, global_metadata.type_count, sizeof(unsigned int)); fflush(stderr);
	build_word_class_counts(cmd_args, word_class_counts, word2class, sent_store_int, global_metadata.line_count, false);

	// Build reverse: <c,v> counts: class followed by word.  This and the normal one are both pretty fast, so no need to parallelize this
	unsigned int * restrict word_class_rev_counts = NULL;
	if (cmd_args.rev_alternate) { // Don't bother building this if it won't be used
		word_class_rev_counts = calloc(1 + cmd_args.num_classes * global_metadata.type_count , sizeof(unsigned int));
		if (word_class_rev_counts == NULL) {
			fprintf(stderr,  "%s: Warning: Unable to allocate enough memory for <v,c>.  %'.1f MB needed.  Falling back to --rev-alternate 0\n", argv_0_basename, ((cmd_args.num_classes * global_metadata.type_count * sizeof(unsigned int)) / (double)1048576 )); fflush(stderr);
			cmd_args.rev_alternate = 0;
		} else {
			memusage += cmd_args.num_classes * global_metadata.type_count * sizeof(unsigned int);
			fprintf(stderr, "%s: Allocating %'.1f MB for word_class_rev_counts: num_classes=%u x type_count=%u x sizeof(uint)=%zu\n", argv_0_basename, (double)(cmd_args.num_classes * global_metadata.type_count * sizeof(unsigned int)) / 1048576 , cmd_args.num_classes, global_metadata.type_count, sizeof(unsigned int)); fflush(stderr);
			build_word_class_counts(cmd_args, word_class_rev_counts, word2class, sent_store_int, global_metadata.line_count, true);
		}

	}

	// Calculate memusage for count_arrays
	for (unsigned char i = 1; i <= cmd_args.max_array; i++) {
		memusage += 2 * (powi(cmd_args.num_classes, i) * sizeof(unsigned int));
		//printf("11 memusage += %zu (now=%zu) count_arrays\n", 2 * (powi(cmd_args.num_classes, i) * sizeof(unsigned int)), memusage);
	}

	clock_t time_model_built = clock();
	if (cmd_args.verbose >= -1)
		fprintf(stderr, "%s: Finished loading %'lu tokens and %'u types (%'u filtered) from %'lu lines in %'.2f CPU secs\n", argv_0_basename, global_metadata.token_count, global_metadata.type_count, number_of_deleted_words, global_metadata.line_count, (double)(time_model_built - time_start)/CLOCKS_PER_SEC); fflush(stderr);
	if (cmd_args.verbose >= -1)
		fprintf(stderr, "%s: Approximate mem usage: %'.1fMB\n", argv_0_basename, (double)memusage / 1048576); fflush(stderr);

	cluster(cmd_args, sent_store_int, global_metadata, word_counts, word_list, word2class, word_bigrams, word_bigrams_rev, word_class_counts, word_class_rev_counts);

	// Now print the final word2class mapping
	if (cmd_args.verbose >= 0) {
		FILE *out_file = stdout;
		if (out_file_string)
			out_file = fopen(out_file_string, "w");
		print_words_and_classes(out_file, global_metadata.type_count, word_list, word_counts, word2class, (int)cmd_args.class_offset);
		fclose(out_file);
	}

	clock_t time_clustered = clock();
	time_t time_t_end;
	time(&time_t_end);
	double time_secs_total = difftime(time_t_end, time_t_start);
	if (cmd_args.verbose >= -1)
		fprintf(stderr, "%s: Finished clustering in %'.2f CPU seconds.  Total wall clock time was about %lim %lis\n", argv_0_basename, (double)(time_clustered - time_model_built)/CLOCKS_PER_SEC, (long)time_secs_total/60, ((long)time_secs_total % 60)  );

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
     --class-offset <c>   Print final word classes starting at a given number (default: %d)\n\
 -h, --help               Print this usage\n\
     --in <file>          Specify input training file (default: stdin)\n\
 -j, --jobs <hu>          Set number of threads to run simultaneously (default: %d threads)\n\
     --min-count <hu>     Minimum count of entries in training set to consider (default: %d occurrences)\n\
     --max-array <c>      Set maximum order of n-grams for which to use an array instead of a sparse hash map (default: %d-grams)\n\
 -n, --num-classes <c>    Set number of word classes (default: square root of vocabulary size)\n\
     --out <file>         Specify output file (default: stdout)\n\
 -q, --quiet              Print less output.  Use additional -q for even less output\n\
     --rev-alternate <u>  How often to alternate using reverse predictive exchange. 0==never, 1==after every normal cycle (default: %u)\n\
     --tune-sents <lu>    Set size of sentence store to tune on (default: first %'lu lines)\n\
     --tune-cycles <hu>   Set max number of cycles to tune on (default: %d cycles)\n\
     --unidirectional     Disable simultaneous bidirectional predictive exchange. Results in faster cycles, but slower & worse convergence\n\
                          If you want to do basic predictive exchange, use --rev-alternate 0 --unidirectional\n\
 -v, --verbose            Print additional info to stderr.  Use additional -v for more verbosity\n\
\n\
", cmd_args.class_offset, cmd_args.num_threads, cmd_args.min_count, cmd_args.max_array, cmd_args.rev_alternate, cmd_args.max_tune_sents, cmd_args.tune_cycles);
}
// -o, --order <i>          Maximum n-gram order in training set to consider (default: %d-grams)\n\
// -w, --weights 'f f ...'  Set class interpolation weights for: 3-gram, 2-gram, 1-gram, rev 2-gram, rev 3-gram. (default: %s)\n\

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
		} else if (!strcmp(argv[arg_i], "--class-offset")) {
			cmd_args->class_offset = (signed char)atoi(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--in")) {
			in_train_file_string = argv[arg_i+1];
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
		} else if (!strcmp(argv[arg_i], "--out")) {
			out_file_string = argv[arg_i+1];
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-q") && strcmp(argv[arg_i], "--quiet"))) {
			cmd_args->verbose--;
		} else if (!strcmp(argv[arg_i], "--rev-alternate")) {
			cmd_args->rev_alternate = (unsigned char) atoi(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--tune-sents")) {
			cmd_args->max_tune_sents = atol(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--tune-cycles")) {
			cmd_args->tune_cycles = (unsigned short) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "--unidirectional"))) {
			cmd_args->unidirectional = true;
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

size_t  sent_buffer2sent_store_int(struct_map_word **ngram_map, char * restrict sent_buffer[restrict], struct_sent_int_info sent_store_int[restrict], const unsigned long num_sents_in_store) {
	size_t local_memusage = 0;

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
		sent_int_temp[0] = map_find_int(ngram_map, "<s>");

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
		sent_int_temp[w_i] = map_find_int(ngram_map, "</s>");

		sentlen_t sent_length = w_i + 1; // Include <s>;  we use this local variable for perspicuity later on
		sent_store_int[i].length = sent_length;

		// Now that we know the actual sentence length, we can allocate the right amount for the sentence
		sent_store_int[i].sent = malloc(sizeof(word_id_t) * sent_length);

		local_memusage += sizeof(word_id_t) * sent_length;

		// Copy the temporary fixed-width array on stack to dynamic-width array in heap
		memcpy(sent_store_int[i].sent, sent_int_temp, sizeof(word_id_t) * sent_length);

		free(sent_i); // Free-up string-based sentence
	}
	return local_memusage;
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
			if (cmd_args.verbose > 3)
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

void tally_class_counts_in_store(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const wclass_t word2class[const], count_arrays_t count_arrays) { // this is a stripped-down version of tally_int_sents_in_store; no temp_class either
	wclass_t class_sent[STDIN_SENT_MAX_WORDS];

	for (unsigned long current_sent_num = 0; current_sent_num < model_metadata.line_count; current_sent_num++) { // loop over sentences
		register sentlen_t sent_length = sent_store_int[current_sent_num].length;

		for (sentlen_t i = 0; i < sent_length; i++) { // loop over words
			class_sent[i] = word2class[ sent_store_int[current_sent_num].sent[i] ];
			//printf("class_sent[%u]=%hu\n", i, class_sent[i]);
			count_arrays[0][  class_sent[i] ]++;
			if (cmd_args.max_array > 1  &&  i > 0) {
				const size_t offset = array_offset(&class_sent[i-1], 2, cmd_args.num_classes);
				count_arrays[1][offset]++;
				//printf("[%hu,%hu]=%u now; offset=%zu\n", class_sent[i-1], class_sent[i], count_arrays[1][offset], offset); fflush(stdout);
				if (cmd_args.max_array > 2  &&  i > 1) {
					const size_t offset = array_offset(&class_sent[i-2], 3, cmd_args.num_classes);
					count_arrays[2][offset]++;
				}
			}
		}
	}
}

void tally_int_sents_in_store(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const wclass_t word2class[const], count_arrays_t count_arrays, const word_id_t temp_word, const wclass_t temp_class) {

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
			fprintf(stderr, "%s: Notice: Truncating pathologically-long line starting with: \"%s %s %s %s %s %s ...\"\n", argv_0_basename, sent_info->sent[1], sent_info->sent[2], sent_info->sent[3], sent_info->sent[4], sent_info->sent[5], sent_info->sent[6]);
			break;
		}

		sent_info->sent[w_i] = pch;
		sent_info->word_lengths[w_i] = strlen(pch);
		//printf("pch=%s; len=%u\n", pch, sent_info->word_lengths[w_i]);

		if (sent_info->word_lengths[w_i] > MAX_WORD_LEN) { // Deal with pathologically-long words
			pch[MAX_WORD_LEN] = '\0';
			sent_info->word_lengths[w_i] = MAX_WORD_LEN;
			fprintf(stderr, "%s: Notice: Truncating pathologically-long word '%s'\n", argv_0_basename, pch);
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

void init_clusters(const struct cmd_args cmd_args, word_id_t vocab_size, wclass_t word2class[restrict], const unsigned int word_counts[const], char * word_list[restrict]) {
	register unsigned long word_i = 0;

	if (cmd_args.class_algo == EXCHANGE) { // It doesn't really matter how you initialize word classes in exchange algo.  This assigns words from the word list an incrementing class number from [0,num_classes-1].  So it's a simple pseudo-randomized initialization.
		register wclass_t class = 0; // [0,num_classes-1]
		for (; word_i < vocab_size; word_i++, class++) {
			if (class == cmd_args.num_classes) // reset
				class = 0;
			//printf("cls=%-4u w_i=%-8lu #(w)=%-8u str(w)=%-20s vocab_size=%u\n", class, word_i, word_counts[word_i], word_list[word_i], vocab_size);
			word2class[word_i] = class;
		}

	} else if (cmd_args.class_algo == BROWN) { // Really simple initialization: one class per word
		for (unsigned long class = 0; word_i < vocab_size; word_i++, class++)
			word2class[word_i] = class;
	}
}

size_t set_bigram_counts(const struct cmd_args cmd_args, struct_word_bigram_entry * restrict word_bigrams, const struct_sent_int_info * const sent_store_int, const unsigned long line_count, const bool reverse) {
	// We first build a hash map of bigrams, since we need random access when traversing the corpus.
	// Then we convert that to an array of linked lists, since we'll need sequential access during the clustering phase of predictive exchange clustering.

	struct_map_bigram *map_bigram = NULL;
	struct_word_bigram bigram;

	for (unsigned long current_sent_num = 0; current_sent_num < line_count; current_sent_num++) { // loop over sentences
		register sentlen_t sent_length = sent_store_int[current_sent_num].length;

		for (sentlen_t i = 1; i < sent_length; i++) { // loop over words in a sentence, starting with the first word after <s>
			if (reverse) {
				bigram.word_2 = sent_store_int[current_sent_num].sent[i-1];
				bigram.word_1 = sent_store_int[current_sent_num].sent[i];
			} else { // Normal direction
				bigram.word_1 = sent_store_int[current_sent_num].sent[i-1];
				bigram.word_2 = sent_store_int[current_sent_num].sent[i];
			}
			map_increment_bigram(&map_bigram, &bigram);
		}
	}

	sort_bigrams(&map_bigram); // really speeds up the next step

	register size_t memusage = 0;
	register word_id_t word_2;
	register word_id_t word_2_last = 0;
	register unsigned int length = 0;
	word_id_t * word_buffer     = malloc(sizeof(word_id_t) * MAX_WORD_PREDECESSORS);
	unsigned int * count_buffer = malloc(sizeof(unsigned int) * MAX_WORD_PREDECESSORS);

	// Iterate through bigram map to get counts of word_2's, so we know how much to allocate for each predecessor list
	struct_map_bigram *entry, *tmp;
	HASH_ITER(hh, map_bigram, entry, tmp) {
		word_2 = (entry->key).word_2;
		//printf("[%u,%u]=%u, w2_last=%u, length=%u\n", (entry->key).word_1, (entry->key).word_2, entry->count, word_2_last, length); fflush(stdout);
		if (word_2 == word_2_last) { // Within successive entry; ie. 2nd entry or greater
			word_buffer[length]  = (entry->key).word_1;
			count_buffer[length] = entry->count;
			length++;
		} else { // New entry; process previous entry
			word_bigrams[word_2_last].length = length;
			word_bigrams[word_2_last].words  = malloc(length * sizeof(word_id_t));
			memcpy(word_bigrams[word_2_last].words,  word_buffer, length * sizeof(word_id_t));
			memusage += length * sizeof(word_id_t);
			word_bigrams[word_2_last].counts = malloc(length * sizeof(unsigned int));
			memcpy(word_bigrams[word_2_last].counts, count_buffer , length * sizeof(unsigned int));
			memusage += length * sizeof(unsigned int);
			//printf("\nword_2_last=%u, length=%u word_1s: ", word_2_last, length);
			//for (unsigned int i = 0; i < length; i++) {
			//	printf("<%u,%u> ", word_bigrams[word_2_last].words[i], word_bigrams[word_2_last].counts[i]);
			//}
			//printf("\n");

			word_2_last = word_2;
			word_buffer[0]  = (entry->key).word_1;
			count_buffer[0] = entry->count;
			length = 1;
		}
	}

	free(word_buffer);
	free(count_buffer);
	delete_all_bigram(&map_bigram);

	return memusage;
}

void build_word_class_counts(const struct cmd_args cmd_args, unsigned int * restrict word_class_counts, const wclass_t word2class[const], const struct_sent_int_info * const sent_store_int, const unsigned long line_count, const bool reverse) {

	for (unsigned long current_sent_num = 0; current_sent_num < line_count; current_sent_num++) { // loop over sentences
		register sentlen_t sent_length = sent_store_int[current_sent_num].length;
		register wclass_t class_i;
		register word_id_t word_id_i_minus_1;

		for (sentlen_t i = 1; i < sent_length; i++) { // loop over words in a sentence, starting with the first word after <s>
			if (reverse) { // Reversed: <c,v>
				class_i           = word2class[sent_store_int[current_sent_num].sent[i-1]];
				word_id_i_minus_1 = sent_store_int[current_sent_num].sent[i];
			} else { // Normal <v,c>
				class_i           = word2class[sent_store_int[current_sent_num].sent[i]];
				word_id_i_minus_1 = sent_store_int[current_sent_num].sent[i-1];
			}
			//printf("i=%hu, sent_len=%u, sent_num=%lu, line_count=%lu, <v,w>=<%u,%u>, <v,c>=<%u,%u>, num_classes=%u, offset=%u (%u * %u + %u), orig_val=%u, rev=%d\n", i, sent_length, current_sent_num, line_count, sent_store_int[current_sent_num].sent[i-1], sent_store_int[current_sent_num].sent[i], word_id_i_minus_1, class_i, cmd_args.num_classes, word_id_i_minus_1 * cmd_args.num_classes + class_i, word_id_i_minus_1, cmd_args.num_classes, class_i, word_class_counts[word_id_i_minus_1 * cmd_args.num_classes + class_i], reverse); fflush(stdout);
			word_class_counts[word_id_i_minus_1 * cmd_args.num_classes + class_i]++;
		}
	}
}

inline float pex_remove_word(const struct cmd_args cmd_args, const struct_model_metadata model_metadata, const word_id_t word, const unsigned int word_count, const wclass_t from_class, wclass_t word2class[], struct_word_bigram_entry * restrict word_bigrams, struct_word_bigram_entry * restrict word_bigrams_rev, unsigned int * restrict word_class_counts, unsigned int * restrict word_class_rev_counts, count_arrays_t count_arrays, const bool is_tentative_move) {
	// See Procedure MoveWord on page 758 of Uszkoreit & Brants (2008):  https://www.aclweb.org/anthology/P/P08/P08-1086.pdf
	const unsigned int count_class = count_arrays[0][from_class];
	const unsigned int new_count_class = count_class - word_count;
	register double delta = count_class * log2(count_class)  -  new_count_class * log2(new_count_class);
	//printf("rm42: word=%u, word_count=%u, from_class=%u, count_class=%u, new_count_class=%u (count_class - word_count), delta=%g\n", word, word_count, from_class, count_class, new_count_class, delta); fflush(stdout);

	if (! is_tentative_move)
		count_arrays[0][from_class] = new_count_class;

	for (unsigned int i = 0; i < word_bigrams[word].length; i++) {
		word_id_t prev_word = word_bigrams[word].words[i];
		//printf(" rm43: i=%u, len=%u, word=%u, offset=%u (prev_word=%u + num_classes=%u * from_class=%u)\n", i, word_bigrams[word].length, word,  (prev_word * cmd_args.num_classes + from_class), prev_word, cmd_args.num_classes, from_class); fflush(stdout);
		const unsigned int word_class_count = word_class_counts[prev_word * cmd_args.num_classes + from_class];
		if (word_class_count != 0) // Can't do log(0)
			delta -= word_class_count * log2(word_class_count);
		const unsigned int new_word_class_count = word_class_count - word_bigrams[word].counts[i];
		delta += new_word_class_count * log2(new_word_class_count);
		//printf(" rm45: word=%u (#=%u), prev_word=%u, #(<v,w>)=%u, from_class=%u, i=%u, count_class=%u, new_count_class=%u, <v,c>=<%u,%u>, #(<v,c>)=%u, new_#(<v,c>)=%u (w-c - %u), delta=%g\n", word, word_count, prev_word, word_bigrams[word].counts[i], from_class, i, count_class, new_count_class, prev_word, from_class, word_class_count, new_word_class_count, word_bigrams[word].counts[i], delta); fflush(stdout);
		//print_word_class_counts(cmd_args, model_metadata, word_class_counts);
		if (! is_tentative_move)
			word_class_counts[prev_word * cmd_args.num_classes + from_class] = new_word_class_count;

	}

	if (cmd_args.rev_alternate && (!is_tentative_move)) { // also update reversed word-class counts
		for (unsigned int i = 0; i < word_bigrams_rev[word].length; i++) {
			const word_id_t next_word = word_bigrams_rev[word].words[i];
			const unsigned int word_class_rev_count = word_class_rev_counts[next_word * cmd_args.num_classes + from_class];
			const unsigned int new_word_class_rev_count = word_class_rev_count - word_bigrams_rev[word].counts[i];
			//printf(" rm47: rev_next_word=%u, rev_#(<v,c>)=%u, rev_new_#(<v,c>)=%u\n", next_word, word_class_rev_count, new_word_class_rev_count); fflush(stdout);
			//print_word_class_counts(cmd_args, model_metadata, word_class_rev_counts);
			word_class_rev_counts[next_word * cmd_args.num_classes + from_class] = new_word_class_rev_count;
		}
	}

	return delta;
}

inline double pex_move_word(const struct cmd_args cmd_args, const word_id_t word, const unsigned int word_count, const wclass_t to_class, wclass_t word2class[], struct_word_bigram_entry * restrict word_bigrams, struct_word_bigram_entry * restrict word_bigrams_rev, unsigned int * restrict word_class_counts, unsigned int * restrict word_class_rev_counts, count_arrays_t count_arrays, const bool is_tentative_move) {
	// See Procedure MoveWord on page 758 of Uszkoreit & Brants (2008):  https://www.aclweb.org/anthology/P/P08/P08-1086.pdf
	unsigned int count_class = count_arrays[0][to_class];
	if (!count_class) // class is empty
		count_class = 1;
	const unsigned int new_count_class = count_class + word_count; // Differs from paper: replace "-" with "+"
	register double delta = count_class * log2(count_class)  -  new_count_class * log2(new_count_class);
	//printf("mv42: word=%u, word_count=%u, to_class=%u, count_class=%u, new_count_class=%u, delta=%g, is_tentative_move=%d\n", word, word_count, to_class, count_class, new_count_class, delta, is_tentative_move); fflush(stdout);

	if (! is_tentative_move)
		count_arrays[0][to_class] = new_count_class;

	for (unsigned int i = 0; i < word_bigrams[word].length; i++) {
		word_id_t prev_word = word_bigrams[word].words[i];
		//printf(" mv43: i=%u, len=%u, word=%u, offset=%u (prev_word=%u + num_classes=%u * to_class=%u)\n", i, word_bigrams[word].length, word,  (prev_word * cmd_args.num_classes + to_class), prev_word, cmd_args.num_classes, to_class); fflush(stdout);
		const unsigned int word_class_count = word_class_counts[prev_word * cmd_args.num_classes + to_class];
		if (word_class_count != 0) { // Can't do log(0)
			if (cmd_args.unidirectional) {
				delta -= (word_class_count * log2(word_class_count));
			} else {
				delta -= (word_class_count * log2(word_class_count)) * 0.6;
			}
		}
		const unsigned int new_word_class_count = word_class_count + word_bigrams[word].counts[i]; // Differs from paper: replace "-" with "+"
		if (new_word_class_count != 0) { // Can't do log(0)
			if (cmd_args.unidirectional) {
				delta += (new_word_class_count * log2(new_word_class_count));
			} else {
				delta += (new_word_class_count * log2(new_word_class_count)) * 0.6;
			}
		}
		//printf(" mv45: word=%u; prev_word=%u, to_class=%u, i=%u, word_count=%u, count_class=%u, new_count_class=%u, <v,c>=<%u,%hu>, #(<v,c>)=%u, new_#(<v,c>)=%u, delta=%g\n", word, prev_word, to_class, i, word_count, count_class, new_count_class, prev_word, to_class, word_class_count, new_word_class_count, delta); fflush(stdout);
		if (! is_tentative_move)
			word_class_counts[prev_word * cmd_args.num_classes + to_class] = new_word_class_count;

	}

	if (cmd_args.rev_alternate) { // also update reversed word-class counts; reversed order of conditionals since the first clause here is more common in this function
		for (unsigned int i = 0; i < word_bigrams_rev[word].length; i++) {
			const word_id_t next_word = word_bigrams_rev[word].words[i];
			const unsigned int word_class_rev_count = word_class_rev_counts[next_word * cmd_args.num_classes + to_class];
			if (word_class_rev_count != 0) // Can't do log(0)
				if (!cmd_args.unidirectional)
					delta -= (word_class_rev_count * log2(word_class_rev_count)) * 0.4;

			const unsigned int new_word_class_rev_count = word_class_rev_count + word_bigrams_rev[word].counts[i];
			if (new_word_class_rev_count != 0) // Can't do log(0)
				if (!cmd_args.unidirectional)
					delta += (new_word_class_rev_count * log2(new_word_class_rev_count)) * 0.4;
			//printf("word=%u, word_class_rev_count=%u, new_word_class_rev_count=%u, delta=%g\n", word, word_class_rev_count, new_word_class_rev_count, delta);
			if (!is_tentative_move)
				word_class_rev_counts[next_word * cmd_args.num_classes + to_class] = new_word_class_rev_count;
		}
	}

	return delta;
}

void cluster(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const unsigned int word_counts[const], char * word_list[restrict], wclass_t word2class[], struct_word_bigram_entry * restrict word_bigrams, struct_word_bigram_entry * restrict word_bigrams_rev, unsigned int * restrict word_class_counts, unsigned int * restrict word_class_rev_counts) {
	unsigned long steps = 0;

	if (cmd_args.class_algo == EXCHANGE) { // Exchange algorithm: See Sven Martin, Jörg Liermann, Hermann Ney. 1998. Algorithms For Bigram And Trigram Word Clustering. Speech Communication 24. 19-37. http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.2354
		// Get initial logprob
		count_arrays_t count_arrays = malloc(cmd_args.max_array * sizeof(void *));
		init_count_arrays(cmd_args, count_arrays);
		tally_class_counts_in_store(cmd_args, sent_store_int, model_metadata, word2class, count_arrays);

		if (cmd_args.verbose > 3) {
			printf("cluster(): 42: "); long unsigned int class_sum=0; for (wclass_t i = 0; i < cmd_args.num_classes; i++) {
				printf("c_%u=%u, ", i, count_arrays[0][i]);
				class_sum += count_arrays[0][i];
			} printf("\nClass Sum=%lu; Corpus Tokens=%lu\n", class_sum, model_metadata.token_count); fflush(stdout);
		}
		double best_log_prob = query_int_sents_in_store(cmd_args, sent_store_int, model_metadata, word_counts, word2class, word_list, count_arrays, -1, 1);

		if (cmd_args.verbose >= -1)
			fprintf(stderr, "%s: Expected Steps:  %'lu (%'u word types x %'u classes x %'u cycles);  initial logprob=%g, PP=%g\n", argv_0_basename, (unsigned long)model_metadata.type_count * cmd_args.num_classes * cmd_args.tune_cycles, model_metadata.type_count, cmd_args.num_classes, cmd_args.tune_cycles, best_log_prob, perplexity(best_log_prob, (model_metadata.token_count - model_metadata.line_count))); fflush(stderr);

		unsigned short cycle = 1; // Keep this around afterwards to print out number of actually-completed cycles
		word_id_t moved_count = 0;
		count_arrays_t temp_count_arrays = malloc(cmd_args.max_array * sizeof(void *));
		init_count_arrays(cmd_args, temp_count_arrays);
		for (; cycle <= cmd_args.tune_cycles; cycle++) {
			const bool is_nonreversed_cycle = (cmd_args.rev_alternate == 0) || (cycle % (cmd_args.rev_alternate+1)); // Only do a reverse predictive exchange (using <c,v>) after every cmd_arg.rev_alternate cycles; if rev_alternate==0 then always do this part.

			clear_count_arrays(cmd_args, temp_count_arrays);
			tally_class_counts_in_store(cmd_args, sent_store_int, model_metadata, word2class, temp_count_arrays);
			double queried_log_prob = query_int_sents_in_store(cmd_args, sent_store_int, model_metadata, word_counts, word2class, word_list, temp_count_arrays, -1, 1);

			if (cmd_args.verbose >= -1) {
				if (is_nonreversed_cycle)
					fprintf(stderr, "%s: Starting normal cycle   ", argv_0_basename);
				else
					fprintf(stderr, "%s: Starting reversed cycle ", argv_0_basename);
				fprintf(stderr, "%-3u with %.2g%% (%u/%u) words exchanged last cycle.    \tlogprob=%g, PP=%g\n", cycle, (100 * (moved_count / (float)model_metadata.type_count)), moved_count, model_metadata.type_count, queried_log_prob, perplexity(queried_log_prob,(model_metadata.token_count - model_metadata.line_count))); fflush(stderr);
			}
			moved_count = 0;

			//#pragma omp parallel for num_threads(cmd_args.num_threads) reduction(+:steps) // non-determinism
			for (word_id_t word_i = 0; word_i < model_metadata.type_count; word_i++) {
			//for (word_id_t word_i = model_metadata.type_count-1; word_i != -1; word_i--) {
				if (cycle < 3 && word_i < cmd_args.num_classes) // don't move high-frequency words in the first (few) iteration(s)
					continue;
				const unsigned int word_i_count = word_counts[word_i];
				const wclass_t old_class = word2class[word_i];
				double scores[cmd_args.num_classes]; // This doesn't need to be private in the OMP parallelization since each thead is writing to different element in the array
				//const double delta_remove_word = pex_remove_word(cmd_args, word_i, word_i_count, old_class, word2class, word_bigrams, word_class_counts, count_arrays, true);
				const double delta_remove_word = 0.0;  // Not really necessary
				const double delta_remove_word_rev = 0.0;  // Not really necessary

				//printf("cluster(): 43: "); long unsigned int class_sum=0; for (wclass_t i = 0; i < cmd_args.num_classes; i++) {
				//	printf("c_%u=%u, ", i, count_arrays[0][i]);
				//	class_sum += count_arrays[0][i];
				//} printf("\nClass Sum=%lu; Corpus Tokens=%lu\n", class_sum, model_metadata.token_count); fflush(stdout);

				#pragma omp parallel for num_threads(cmd_args.num_threads) reduction(+:steps)
				for (wclass_t class = 0; class < cmd_args.num_classes; class++) { // class values range from 0 to cmd_args.num_classes-1
					//if (old_class == class) {
					//	scores[class] = 0.0;
					//	continue;
					//}

					if (is_nonreversed_cycle) {
						scores[class] = delta_remove_word + pex_move_word(cmd_args, word_i, word_i_count, class, word2class, word_bigrams, word_bigrams_rev, word_class_counts, word_class_rev_counts, count_arrays, true);
					} else { // This is the reversed one
						scores[class] = delta_remove_word_rev + pex_move_word(cmd_args, word_i, word_i_count, class, word2class, word_bigrams_rev, word_bigrams, word_class_rev_counts, word_class_counts, count_arrays, true);
					}
					steps++;
				}

				const wclass_t best_hypothesis_class = which_max(scores, cmd_args.num_classes);
				const double best_hypothesis_score = max(scores, cmd_args.num_classes);

				if (cmd_args.verbose > 1) {
					printf("Orig score for word w_«%u» using class «%hu» is %g;  Hypos %u-%u: ", word_i, old_class, scores[old_class], 1, cmd_args.num_classes);
					fprint_array(stdout, scores, cmd_args.num_classes, ","); fflush(stdout);
					//if (best_hypothesis_score > 0) { // Shouldn't happen
					//	fprintf(stderr, "Error: best_hypothesis_score=%g for class %hu > 0\n", best_hypothesis_score, best_hypothesis_class); fflush(stderr);
					//	exit(9);
					//}
				}

				//if (scores[old_class] > best_hypothesis_score) { // We've improved
				if (old_class != best_hypothesis_class) { // We've improved
					moved_count++;

					if (cmd_args.verbose > 0)
						fprintf(stderr, " Moving id=%-7u count=%-7u %-18s %u -> %u\t(%g -> %g)\n", word_i, word_counts[word_i], word_list[word_i], old_class, best_hypothesis_class, scores[old_class], best_hypothesis_score); fflush(stderr);
					//word2class[word_i] = best_hypothesis_class;
					word2class[word_i] = best_hypothesis_class;
					if (isnan(best_hypothesis_score)) { // shouldn't happen
						fprintf(stderr, "Error: best_hypothesis_score=%g :-(\n", best_hypothesis_score); fflush(stderr);
						exit(5);
					} else {
						best_log_prob += best_hypothesis_score;
					}

					if (is_nonreversed_cycle) {
						pex_remove_word(cmd_args, model_metadata, word_i, word_i_count, old_class, word2class, word_bigrams, word_bigrams_rev, word_class_counts, word_class_rev_counts, count_arrays, false);
						pex_move_word(cmd_args, word_i, word_i_count, best_hypothesis_class, word2class, word_bigrams, word_bigrams_rev, word_class_counts, word_class_rev_counts, count_arrays, false);
					} else { // This is the reversed one
						pex_remove_word(cmd_args, model_metadata, word_i, word_i_count, old_class, word2class, word_bigrams_rev, word_bigrams, word_class_rev_counts, word_class_counts, count_arrays, false);
						pex_move_word(cmd_args, word_i, word_i_count, best_hypothesis_class, word2class, word_bigrams_rev, word_bigrams,  word_class_rev_counts, word_class_counts, count_arrays, false);
					}
				}
			}

			// In principle if there's no improvement in the determinitistic exchange algo, we can stop cycling; there will be no more gains
			if (!moved_count) // Nothing moved in last cycle, so that's it
				break;
		}

		free_count_arrays(cmd_args, temp_count_arrays);
		free(temp_count_arrays);
		free_count_arrays(cmd_args, count_arrays);
		free(count_arrays);

		if (cmd_args.verbose >= -1)
			fprintf(stderr, "%s: Completed steps: %'lu (%'u word types x %'u classes x %'u cycles);\n", argv_0_basename, steps, model_metadata.type_count, cmd_args.num_classes, cycle-1); fflush(stderr);
			//fprintf(stderr, "%s: Completed steps: %'lu (%'u word types x %'u classes x %'u cycles);     best logprob=%g, PP=%g\n", argv_0_basename, steps, model_metadata.type_count, cmd_args.num_classes, cycle-1, best_log_prob, perplexity(best_log_prob,(model_metadata.token_count - model_metadata.line_count))); fflush(stderr);

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


double query_int_sents_in_store(const struct cmd_args cmd_args, const struct_sent_int_info * const sent_store_int, const struct_model_metadata model_metadata, const unsigned int word_counts[const], const wclass_t word2class[const], char * word_list[restrict], const count_arrays_t count_arrays, const word_id_t temp_word, const wclass_t temp_class) {
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
			const unsigned int class_i_count = count_arrays[0][class_i];
			//float word_i_count_for_next_freq_score = word_i_count ? word_i_count : 0.2; // Using a very small value for unknown words messes up distribution
			if (cmd_args.verbose > 3) {
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
			float weights_class[] = {0.4, 0.16, 0.01, 0.1, 0.33};
			//float weights_class[] = {0.0, 0.0, 1.0, 0.0, 0.0};
			//float weights_class[] = {0.0, 0.99, 0.01, 0.0, 0.0};
			//float weights_class[] = {0.8, 0.19, 0.01, 0.0, 0.0};
			//float weights_class[] = {0.69, 0.15, 0.01, 0.15, 0.0};
			float order_probs[5] = {0};
			order_probs[2] = class_i_count / (float)model_metadata.token_count; // unigram probs
			float sum_weights = weights_class[2]; // unigram prob will always occur
			float sum_probs = weights_class[2] * order_probs[2]; // unigram prob will always occur

			//const float transition_prob = class_ngram_prob(cmd_args, count_arrays, class_map, i, class_i, class_i_count, class_sent, CLASSLEN, model_metadata, weights_class);
			if ((cmd_args.max_array > 2) && (i > 1)) { // Need at least "<s> w_1" in history
				order_probs[0] = count_arrays[2][ array_offset(&class_sent[i-2], 3, cmd_args.num_classes) ] / (float)count_arrays[1][ array_offset(&class_sent[i-1], 2, cmd_args.num_classes) ]; // trigram probs
				order_probs[0] = isnan(order_probs[0]) ? 0.0f : order_probs[0]; // If the bigram history is 0, result will be a -nan
				sum_weights += weights_class[0];
				sum_probs += weights_class[0] * order_probs[0];
			} else {
				weights_class[0] = 0.0;
			}

			// We'll always have at least "<s>" in history.  And we'll always have Vienna.
			order_probs[1] = count_arrays[1][ array_offset(&class_sent[i-1], 2, cmd_args.num_classes) ] / (float)count_arrays[0][ array_offset(&class_sent[i], 1, cmd_args.num_classes) ]; // bigram probs
			//printf("order_probs[1] = %u / %u; [%hu,%hu] \n", count_arrays[1][ array_offset(&class_sent[i], 2, cmd_args.num_classes) ], count_arrays[0][ array_offset(&class_sent[i], 1, cmd_args.num_classes)], class_sent[i-1], class_sent[i]);
			sum_weights += weights_class[1];
			sum_probs += weights_class[1] * order_probs[1];

			if (i < sent_length-1) { // Need at least "</s>" to the right
				order_probs[3] = count_arrays[1][ array_offset(&class_sent[i], 2, cmd_args.num_classes) ] / (float)count_arrays[0][ array_offset(&class_sent[i+1], 1, cmd_args.num_classes) ]; // future bigram probs
				sum_weights += weights_class[3];
				sum_probs += weights_class[3] * order_probs[3];
			}

			if ((cmd_args.max_array > 2) && (i < sent_length-2)) { // Need at least "w </s>" to the right
			order_probs[4] = count_arrays[2][ array_offset(&class_sent[i], 3, cmd_args.num_classes) ] / (float)count_arrays[1][ array_offset(&class_sent[i+1], 2, cmd_args.num_classes) ]; // future trigram probs
			order_probs[4] = isnan(order_probs[4]) ? 0.0f : order_probs[4]; // If the bigram history is 0, result will be a -nan
				sum_weights += weights_class[4];
				sum_probs += weights_class[4] * order_probs[4];
			} else {
				weights_class[4] = 0.0;
			}

			const float transition_prob = sum_probs / sum_weights;
			const float class_prob = emission_prob * transition_prob;


			if (cmd_args.verbose > 2) {
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

void clear_count_arrays(const struct cmd_args cmd_args, count_arrays_t count_arrays) {
	for (unsigned char i = 1; i <= cmd_args.max_array; i++) { // Start with unigrams in count_arrays[0], ...
		memset(count_arrays[i-1], 0, powi(cmd_args.num_classes, i) * sizeof(unsigned int)); // powi() is in clustercat-math.c
	}
}

void free_count_arrays(const struct cmd_args cmd_args, count_arrays_t count_arrays) {
	for (unsigned char i = 1; i <= cmd_args.max_array; i++) { // Start with unigrams in count_arrays[0], ...
		free(count_arrays[i-1]);
	}
}

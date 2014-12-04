/** Induces word categories
 *  By Jon Dehdari, 2014
 *  Usage: ./clustercat [options] < corpus.tok.txt > classes.tsv
**/

#include <limits.h>				// UCHAR_MAX, UINT_MAX
#include <float.h>				// DBL_MAX, etc.
#include <time.h>				// clock_t, clock(), CLOCKS_PER_SEC
#include <stdbool.h>

#include "clustercat.h"				// Model importing/exporting functions
#include "clustercat-array.h"		// which_maxf()
#include "clustercat-data.h"
#include "clustercat-io.h"			// fill_sent_buffer()

#define USAGE_LEN 10000
#define SENT_BUF_LEN 10000 // Not a very interesting value

// Declarations
void get_usage_string(char * restrict usage_string, int usage_len);
void parse_cmd_args(const int argc, char **argv, char * restrict usage, struct cmd_args *cmd_args);
void free_sent_info(struct_sent_info sent_info);
char * restrict class_algo = NULL;

struct_map *ngram_map       = NULL;	// Must initialize to NULL
struct_map_class *class_map = NULL;	// Must initialize to NULL
struct_map_word_class *word2class_map = NULL;	// Must initialize to NULL
DECLARE_DATA_STRUCT_FLOAT; // for word_word_float_map
char usage[USAGE_LEN];


// Defaults
struct cmd_args cmd_args = {
	.class_algo             = EXCHANGE,
	.dev_file               = NULL,
	.max_tune_sents         = 10000,
	.min_count              = 0,
	.class_order            = 3,
	.num_threads            = 6,
	.num_classes            = 100,
	.tune_cycles            = 50,
	.verbose                = 0,
};



int main(int argc, char **argv) {
	clock_t time_start = clock();
	argv_0_basename = basename(argv[0]);
	get_usage_string(usage, USAGE_LEN); // This is a big scary string, so build it elsewhere

	//printf("sizeof(cmd_args)=%zd\n", sizeof(cmd_args));
	parse_cmd_args(argc, argv, usage, &cmd_args);

	struct_model_metadata global_metadata;
	global_metadata.token_count = 0;
	global_metadata.line_count  = 0;


	// The list of unique words should always include <s>, unknown word, and </s>
	map_update_entry(&ngram_map, "<s>", 0);
	map_update_entry(&ngram_map, UNKNOWN_WORD, 0);
	map_update_entry(&ngram_map, "</s>", 0);

	char * restrict sent_buffer[SENT_BUF_LEN];
	char * restrict sent_store[cmd_args.max_tune_sents];
	unsigned long num_sents_in_buffer = 0; // We might need this number later if a separate dev set isn't provided;  we'll just tune on final buffer.
	unsigned long num_sents_in_store = 0;
	while (1) {
		// Fill sentence buffer
		num_sents_in_buffer = fill_sent_buffer(stdin, sent_buffer, SENT_BUF_LEN);
		//printf("cmd_args.max_tune_sents=%lu; global_metadata.line_count=%lu; num_sents_in_buffer=%lu\n", cmd_args.max_tune_sents, global_metadata.line_count, num_sents_in_buffer);
		if ((num_sents_in_buffer == 0) || ( cmd_args.max_tune_sents <= global_metadata.line_count)) // No more sentences in buffer
			break;

		global_metadata.line_count  += num_sents_in_buffer;
		global_metadata.token_count += process_sents_in_buffer(sent_buffer, num_sents_in_buffer, &ngram_map, &class_map, true, false);
		num_sents_in_store += copy_buffer_to_store(sent_buffer, num_sents_in_buffer, sent_store, num_sents_in_store, cmd_args.max_tune_sents ); // Separate from process_sents_in_buffer() since we call that function in two separate contexts
	}


	clock_t time_model_built = clock();
	fprintf(stderr, "%s: Finished loading %lu tokens from %lu lines in %.2f secs\n", argv_0_basename, global_metadata.token_count, global_metadata.line_count, (double)(time_model_built - time_start)/CLOCKS_PER_SEC);
	unsigned long vocab_size      = map_count(&ngram_map);
	//unsigned long class_entries = map_print_entries(&class_map, "#CL ", PRIMARY_SEP_CHAR, 0);
	unsigned long ngram_entries   = map_count(&ngram_map);
	unsigned long total_entries   = vocab_size + ngram_entries;
	fprintf(stderr, "  %lu entries:  %lu types,  %lu word ngrams\n", total_entries, vocab_size, ngram_entries);
	unsigned long map_entries = vocab_size + ngram_entries;
	fprintf(stderr, "%s: Approximate mem usage:  maps: %lu x %zu = %lu; total: %.1fMB\n", argv_0_basename, map_entries, sizeof(struct_map), sizeof(struct_map) * map_entries, (double)((sizeof(struct_map) * map_entries)) / 1048576);

	if (vocab_size <= cmd_args.num_classes) {
		fprintf(stderr, "%s: Error: Number of classes (%u) is not less than vocabulary size (%lu).  Decrease the value of --num-classes\n", argv_0_basename, cmd_args.num_classes, vocab_size);
		exit(3);
	}

	// Get list of unique words
	char **unique_words = (char **)malloc(vocab_size * sizeof(char*));
	get_keys(&ngram_map, unique_words);

	init_clusters(cmd_args, vocab_size, unique_words, &word2class_map);
	clock_t time_clusters_initialized = clock();
	if (cmd_args.verbose > 0) {
		fprintf(stderr, "%s: Finished initializing clusters in %.2f secs\n", argv_0_basename, (double)(time_clusters_initialized - time_model_built)/CLOCKS_PER_SEC);
		fflush(stderr);
	}

	cluster(cmd_args, sent_store, num_sents_in_store, vocab_size, unique_words, &ngram_map, &word2class_map);

	clock_t time_clustered = clock();
	fprintf(stderr, "%s: Finished clustering in %.2f secs\n", argv_0_basename, (double)(time_clustered - time_model_built)/CLOCKS_PER_SEC);

	free(unique_words);
	exit(0);
}


void get_usage_string(char * restrict usage_string, int usage_len) {

	snprintf(usage_string, usage_len, "ClusterCat  (c) 2014 Jon Dehdari - LGPL v3 or Apache v2\n\
\n\
Usage:    clustercat [options] < corpus.tok.txt > classes.tsv \n\
\n\
Function: Induces word categories from plaintext\n\
\n\
Options:\n\
     --class-algo <s>     Set class-induction algorithm {brown,exchange} (default: exchange)\n\
     --dev-file <file>    Use separate file to tune on (default: training set, from stdin)\n\
 -h, --help               Print this usage\n\
 -j, --jobs <i>           Set number of threads to run simultaneously (default: %d threads)\n\
     --min-count <i>      Minimum count of entries in training set to consider (default: %d occurrences)\n\
 -n, --num-classes <i>    Set number of word classes (default: %d classes)\n\
     --tune-sents <i>     Set size of sentence store to tune on (default: first %lu sentences)\n\
     --tune-cycles <i>    Set max number of cycles to tune on (default: %d cycles)\n\
 -v, --verbose            Print additional info to stderr.  Use additional -v for more verbosity\n\
\n\
", cmd_args.num_threads, cmd_args.min_count, cmd_args.num_classes, cmd_args.max_tune_sents, cmd_args.tune_cycles);
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
		} else if (!strcmp(argv[arg_i], "--dev-file")) {
			cmd_args->dev_file = argv[arg_i+1];
			printf("Bug Jon to implement --dev-file!\n");
			exit(1);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-j") && strcmp(argv[arg_i], "--jobs"))) {
			cmd_args->num_threads = (unsigned int) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--min-count")) {
			cmd_args->min_count = (unsigned int) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-n") && strcmp(argv[arg_i], "--num-classes"))) {
			cmd_args->num_classes = (wclass_t) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-o") && strcmp(argv[arg_i], "--order"))) {
			cmd_args->class_order = (unsigned char) atoi(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--tune-sents")) {
			cmd_args->max_tune_sents = atol(argv[arg_i+1]);
			arg_i++;
		} else if (!strcmp(argv[arg_i], "--tune-cycles")) {
			cmd_args->tune_cycles = (unsigned short) atol(argv[arg_i+1]);
			arg_i++;
		} else if (!(strcmp(argv[arg_i], "-v") && strcmp(argv[arg_i], "--verbose"))) {
			cmd_args->verbose++;
		} else if (!strncmp(argv[arg_i], "-", 1)) { // Unknown flag
			printf("%s: Unknown command-line argument: %s\n\n", argv_0_basename, argv[arg_i]);
			printf("%s", usage);
			exit(2);
		}
	}
}

void increment_ngram_variable_width(struct_map **ngram_map, char * restrict sent[const], const short * restrict word_lengths, short start_position, const sentlen_t i) {
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
		map_increment_entry(ngram_map, jp);
		//if (diff > 0) // 0 allows for unigrams
			jp += sizeof_char + word_lengths[j];
	}
}

void increment_ngram_fixed_width(struct_map_class **map, wclass_t sent[const], short start_position, const sentlen_t i) {
	size_t sizeof_wclass = sizeof(wclass_t);
	unsigned char ngram_len = i - start_position + 1;

	wclass_t ngram[ngram_len + (CLASSLEN - 1)]; // We reserve more space to allow for eg. the final unigram to be padded with zeros afterwards, since a fixed-width ngram will be passed-on to the map.
	memset(ngram, 0, ngram_len + (CLASSLEN - 1));
	memcpy(&ngram, &sent[start_position], ngram_len);
	printf("ngm++: ngmlen=%u, ngram=[%hu,%hu,%hu,%hu,%hu]\n", ngram_len + (CLASSLEN - 1), ngram[0], ngram[1], ngram[2], ngram[3], ngram[4]);

	wclass_t * restrict jp = ngram;
	for (sentlen_t j = start_position; j <= i; ++j, --ngram_len) { // Traverse longest n-gram string
		//if (cmd_args.verbose > 1)
			//printf("increment_ngram4: start_position=%d, i=%i, w_i=%hu, ngram_len=%d, ngram=<<%hu,%hu,%hu>>, jp=<<%hu,%hu,%hu,%hu>>\n", start_position, i, sent[i], ngram_len, ngram[0], ngram[1], ngram[2], jp[0], jp[1], jp[2], jp[3]);
		map_increment_entry_fixed_width(map, jp);
		jp += sizeof_wclass;
	}
}

unsigned long copy_buffer_to_store(char * restrict sent_buffer[const], const unsigned long num_sents_in_buffer, char * restrict sent_store[], unsigned long num_sents_in_store, const unsigned long max_tune_sents) {
	for (unsigned long i = 0; (num_sents_in_store <= max_tune_sents) && (i < num_sents_in_buffer); i++, num_sents_in_store++) {
		if (sent_buffer[i] == NULL) // The last bit of the buffer might be empty
			break;
		sent_store[i] = sent_buffer[i];
	}
	return num_sents_in_store;
}

unsigned long process_sents_in_buffer(char * restrict sent_buffer[], const unsigned long num_sents_in_buffer, struct_map **ngram_map, struct_map_class **class_map, bool count_word_ngrams, bool count_class_ngrams) {
	unsigned long token_count = 0;
	unsigned long current_sent_num;
	char local_sent_copy[STDIN_SENT_MAX_CHARS];
	local_sent_copy[STDIN_SENT_MAX_CHARS-1] = '\0'; // Ensure at least last element of array is terminating character

	if (cmd_args.verbose > 2) // Precede program basename to verbose notices
		fprintf(stderr, "%s: L=lines; W=words\t", argv_0_basename);

	//#pragma omp parallel for private(current_sent_num) private(local_sent_copy) reduction(+:token_count) num_threads(cmd_args.num_threads) // static < dynamic < runtime <= auto < guided
	for (current_sent_num = 0; current_sent_num < num_sents_in_buffer; current_sent_num++) {
		strncpy(local_sent_copy, sent_buffer[current_sent_num], STDIN_SENT_MAX_WORDS-2); // Strtok, which is used later, is destructive

		if (cmd_args.verbose > 1 && count_class_ngrams)
			printf("sent_buffer[%lu]: <<%s>>\n", current_sent_num, local_sent_copy);
		token_count += process_sent(local_sent_copy, ngram_map, class_map, count_word_ngrams, count_class_ngrams);
		//if (count_class_ngrams)
			//printf("42: count_word_ngrams: %i; count_class_ngrams: %i\n", count_word_ngrams, count_class_ngrams); fflush(stdout);
		if (cmd_args.verbose > 2 && (current_sent_num % 1000000 == 0) && (current_sent_num > 0))
			fprintf(stderr, "%liL/%luW ", current_sent_num, token_count); fflush(stderr);
	}

	if (cmd_args.verbose > 2) // Add final newline to verbose notices
		fprintf(stderr, "\n"); fflush(stderr);

	return token_count;
}

unsigned long process_sent(char * restrict sent_str, struct_map **ngram_map, struct_map_class **class_map, bool count_word_ngrams, bool count_class_ngrams) {
	if (!strncmp(sent_str, "\n", 1)) // Ignore empty lines
		return 0;

	struct_sent_info sent_info = {0};
	sent_info.sent = (char **)malloc(STDIN_SENT_MAX_WORDS * sizeof(char*));

	// We could have built up the word n-gram counts directly from sent_str, but it's
	// the only one out of the three models we're building that we can do this way, and
	// it's simpler to have a more uniform way of building these up.

	tokenize_sent(sent_str, &sent_info, count_word_ngrams);
	unsigned long token_count = sent_info.length;
	if (cmd_args.verbose > 1 && count_class_ngrams) {
		printf("sent_str: <<%s>>\n", sent_str);
		print_sent_info(&sent_info);
	}

	// In the following loop we interpret i in two different ways.  For word/class n-gram models,
	// it's the right-most word in the n-gram. I wrote increment_ngram() earlier using the right-most interpretation of i.
	register sentlen_t i;
	for (i = 0; i < sent_info.length; i++) {

		if (count_word_ngrams)
			increment_ngram_variable_width(ngram_map, sent_info.sent, sent_info.word_lengths, i, i); // N-grams starting point is 0, for <s>;  We only need unigrams for visible words
			//printf("incrementing w=%s to %u (inter alia)\n", sent_info.sent[i], map_find_entry(ngram_map, sent_info.sent[i]));
		if (count_class_ngrams && cmd_args.class_order) {
			sentlen_t start_position_class = (i >= cmd_args.class_order-1) ? i - (cmd_args.class_order-1) : 0; // N-grams starting point is 0, for <s>
			//printf("i: %u, sent_len=%u\t", i, sent_info.length);
			//if (i - start_position_class > 1)
			//	printf("w_i-2=%s (cls: %hu)\t", sent_info.sent[i-1], sent_info.class_sent[i-1]);
			//if (i - start_position_class > 0)
			//	printf("w_i-1=%s (cls: %hu)\t", sent_info.sent[i-1], sent_info.class_sent[i-1]);
			//printf("w_i=%s (cls: %hu)\n", sent_info.sent[i], sent_info.class_sent[i]);
			increment_ngram_fixed_width(class_map, sent_info.class_sent, start_position_class, i);
		}
	}

	free(sent_info.sent);
	return token_count;
}


void tokenize_sent(char * restrict sent_str, struct_sent_info *sent_info, bool count_word_ngrams) {
	// Stupid strtok is destructive
	char * restrict pch = NULL;
	pch = strtok(sent_str, TOK_CHARS);

	// Initialize first element in sentence to <s>
	sent_info->sent[0] = "<s>";
	sent_info->class_sent[0] = get_class(&word2class_map, "<s>", UNKNOWN_WORD_CLASS);
	//sent_info->word_lengths[0]  = strlen("<s>");
	sent_info->sent_counts[0] = map_find_entry(&ngram_map, "<s>");

	sentlen_t w_i = 1; // Word 0 is <s>

	for (; pch != NULL  &&  w_i < SENT_LEN_MAX; w_i++) {
		if (w_i == STDIN_SENT_MAX_WORDS - 1) { // Deal with pathologically-long lines
			fprintf(stderr, "%s: Warning: Truncating pathologically-long line starting with: %s %s %s %s %s %s ...\n", argv_0_basename, sent_info->sent[1], sent_info->sent[2], sent_info->sent[3], sent_info->sent[4], sent_info->sent[5], sent_info->sent[6]);
			break;
		}

		sent_info->sent[w_i] = pch;
		if (count_word_ngrams)
			sent_info->word_lengths[w_i] = strlen(pch);
			sent_info->sent_counts[w_i] = map_find_entry(&ngram_map, pch);
		if (!count_word_ngrams)
			sent_info->class_sent[w_i] =  get_class(&word2class_map, pch, UNKNOWN_WORD_CLASS);

		if (sent_info->word_lengths[w_i] > MAX_WORD_LEN) { // Deal with pathologically-long words
			pch[MAX_WORD_LEN] = '\0';
			sent_info->word_lengths[w_i] = MAX_WORD_LEN;
			fprintf(stderr, "%s: Warning: Truncating pathologically-long word '%s'\n", argv_0_basename, pch);
		}

		pch = strtok(NULL, TOK_CHARS);
	}

	// Initialize last element in sentence to </s>
	sent_info->sent[w_i] = "</s>";
	sent_info->class_sent[w_i] = get_class(&word2class_map, "</s>", UNKNOWN_WORD_CLASS);
	sent_info->sent_counts[w_i] = map_find_entry(&ngram_map, "</s>");
	//sent_info->word_lengths[w_i]  = strlen("</s>");
	sent_info->length = w_i + 1; // Include <s>
	if (cmd_args.verbose > 1)
		printf("88sent_str: count_word_ngrams=%i; <<%s>>\n", count_word_ngrams, sent_str);
}

// Slightly different from free_sent_info() since we don't free the individual words in sent_info.sent here
void free_sent_info(struct_sent_info sent_info) {
	for (sentlen_t i = 1; i < sent_info.length-1; ++i) // Assumes word_0 is <s> and word_sentlen is </s>, which weren't malloc'd
		free(sent_info.sent[i]);

	free(sent_info.sent);
}

void init_clusters(const struct cmd_args cmd_args, unsigned long vocab_size, char **unique_words, struct_map_word_class **word2class_map) {
	register unsigned long word_i = 0;

	if (cmd_args.class_algo == EXCHANGE) { // It doesn't really matter how you initialize word classes in exchange algo.  This assigns words from the word list an incrementing class number from [0,num_classes].  So it's a simple pseudo-randomized initialization.
		register wclass_t class = 1; // 0 is reserved
		for (; word_i < vocab_size; word_i++, class++) {
			if (class >= cmd_args.num_classes)
				class = 1;
			//printf("class=%u, word=%s, word_i=%lu, vocab_size=%lu\n", class, unique_words[word_i], word_i, vocab_size);
			map_update_class(word2class_map, unique_words[word_i], class);
		}

	} else if (cmd_args.class_algo == BROWN) { // Really simple initialization: one class per word
		for (unsigned long class = 0; word_i < vocab_size; word_i++, class++)
			map_update_class(word2class_map, unique_words[word_i], class);
	}
}

void cluster(const struct cmd_args cmd_args, char * restrict sent_store[const], unsigned long num_sents_in_store, unsigned long vocab_size, char **unique_words, struct_map **ngram_map, struct_map_word_class **word2class_map) {

	unsigned long steps = 0;

	if (cmd_args.class_algo == EXCHANGE) { // Exchange algorithm: See Sven Martin, Jörg Liermann, Hermann Ney. 1998. Algorithms For Bigram And Trigram Word Clustering. Speech Communication 24. 19-37. http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.2354
		for (unsigned short cycle = 0; cycle < cmd_args.tune_cycles; cycle++) {
			for (unsigned long word_i = 0; word_i < vocab_size; word_i++) {
				char * restrict word = unique_words[word_i];
				float best_log_prob = FLT_MIN;
				float log_probs[cmd_args.num_classes];
				#pragma omp parallel for num_threads(cmd_args.num_threads) reduction(+:steps)
				for (wclass_t class = 0; class < cmd_args.num_classes; class++) {
					steps++;
					// Get log prob
					struct_map_class *class_map = NULL; // Build local counts of classes, for flexibility
					process_sents_in_buffer(sent_store, num_sents_in_store, ngram_map, &class_map, false, true); // Get class ngram counts
					log_probs[class] = query_sents_in_store(cmd_args, sent_store, num_sents_in_store, ngram_map, &class_map, word2class_map);
				}
				wclass_t best_class = which_maxf(log_probs, cmd_args.num_classes);
				if (best_log_prob < maxf(log_probs, cmd_args.num_classes))
					printf("Moving '%s' to class %u\n", word, best_class);
				else
					break; // Moving stuff around didn't help, so we're done
			}
		}
		printf("steps: %lu (%lu words x %u classes x %u cycles)\n", steps, vocab_size, cmd_args.num_classes, cmd_args.tune_cycles);

	} else if (cmd_args.class_algo == BROWN) { // Agglomerative clustering.  Stops when the number of current clusters is equal to the desired number in cmd_args.num_classes
		// "Things equal to nothing else are equal to each other." --Anon
		for (unsigned long current_num_classes = vocab_size; current_num_classes > cmd_args.num_classes; current_num_classes--) {
			for (unsigned long word_i = 0; word_i < vocab_size; word_i++) {
				char * restrict word = unique_words[word_i];
				float log_probs[cmd_args.num_classes];
				//#pragma omp parallel for num_threads(cmd_args.num_threads)
				for (wclass_t class = 0; class < cmd_args.num_classes; class++, steps++) {
					// Get log prob
					log_probs[class] = -1 * (class+1); // Dummy predicate
				}
				wclass_t best_class = which_maxf(log_probs, cmd_args.num_classes);
				printf("Moving '%s' to class %u\n", word, best_class);
			}
		}
	}
}


struct_sent_info parse_input_line(char * restrict line_in, struct_map **ngram_map) {
	// Make local copy of sentence, for threadsafe processing
	struct_sent_info sent_info;
	sent_info.sent = (char **)malloc(STDIN_SENT_MAX_WORDS * sizeof(char*));
	sent_info.sent[0] = "<s>";
	//sent_info.word_lengths[0] = strlen("<s>");
	sent_info.sent_counts[0]  = map_find_entry(ngram_map, "<s>");
	sent_info.class_sent[0]   = get_class(&word2class_map, "<s>", UNKNOWN_WORD_CLASS);

	sentlen_t i;
	char * restrict pch;

	for (i = 1, pch = line_in; i < SENT_LEN_MAX ; i++) { // Tokenize & save sentence input from stdin
		sentlen_t toklen = strcspn(pch, " \n\t");

		if (toklen == 0) { // End of sentence
			sent_info.sent[i] = "</s>";
			sent_info.class_sent[i]   = get_class(&word2class_map, "</s>", UNKNOWN_WORD_CLASS);
			//sent_info.word_lengths[i] = strlen("</s>"); // We'll need this several times later, for memory allocation
			sent_info.sent_counts[i]  = map_find_entry(ngram_map, "</s>");
			break;
		}

		sent_info.sent[i] = malloc(toklen+1);
		strncpy(sent_info.sent[i], pch, toklen); // Threadsafe copy doesn't touch original
		sent_info.sent[i][toklen] = '\0';

		sent_info.sent_counts[i] = map_find_entry(ngram_map, sent_info.sent[i]);
		//sent_info.word_lengths[i]  = toklen; // We'll need this several times later, for memory allocation

		sent_info.class_sent[i] = get_class(&word2class_map, sent_info.sent[i], UNKNOWN_WORD_CLASS);
		//class = map_find_entry(&model_maps.class_map, class) ? class : UNKNOWN_WORD_CLASS; // If count of class is 0, then reassign it to the unknown class

		pch += toklen+1;

		if (cmd_args.verbose > 2)
			printf("line=%u i=%d\twlen=%d\tcnt=%d\tcls=%u\tw=%s\n", __LINE__, i, toklen, sent_info.sent_counts[i], sent_info.class_sent[i], sent_info.sent[i]);
	}
	sent_info.length = i;

	return sent_info;
}


float query_sents_in_store(const struct cmd_args cmd_args, char * restrict sent_store[const], const unsigned long num_sents_in_store, struct_map **ngram_map, struct_map_class **class_map, struct_map_word_class **word2class_map) {
	float sum_log_probs = 0.0; // For perplexity calculation

	unsigned long current_sent_num;
	// Ensure that the printf statement for actually printing the final sentence query is preceded by an omp ordered pragma construct
	//#pragma omp parallel for private(current_sent_num) num_threads(cmd_args.num_threads) reduction(+:sum_log_probs)
	for (current_sent_num = 0; current_sent_num < num_sents_in_store; current_sent_num++) {

		char * restrict current_sent = sent_store[current_sent_num];
		//struct_sent_info parse_input_line(char * restrict line_in, const struct_sent_info sent_info_a, struct_map **ngram_map) {
		struct_sent_info sent_info = parse_input_line(current_sent, ngram_map);
		if (cmd_args.verbose > 2)
			print_sent_info(&sent_info);

		float sent_score = 0.0; // Initialize with identity element

		for (sentlen_t i = 1; i <= sent_info.length; i++) {
			char * restrict word_i = sent_info.sent[i];
			const wclass_t * class_i = &sent_info.class_sent[i];
			wclass_t class_i_entry[CLASSLEN] = {USHRT_MAX};
			class_i_entry[0] = *class_i;
			const unsigned int word_i_count = sent_info.sent_counts[i];
			const unsigned int class_i_count = map_find_entry_fixed_width(class_map, class_i_entry);
			//float word_i_count_for_next_freq_score = word_i_count ? word_i_count : 0.2; // Using a very small value for unknown words messes up distribution
			if (cmd_args.verbose > 0)
				printf("line=%u i=%d\tcnt=%d\tcls=%u\tcls_cnt=%d\tcls_entry=[%hu,%hu,%hu]\tw=%s\n", __LINE__, i, word_i_count, *class_i, class_i_count, class_i_entry[0], class_i_entry[1], class_i_entry[2], word_i);

#if 0
			// Class N-gram Prob
			float the_class_prob = 0.0;
			if (weights.interpolation[CLASS] != 0.0) { // Nonexistent class info in model yields nan's, which taints interpolated probs
				// Class prob is transition prob * emission prob
				float emission_prob = word_i_count ? (float)word_i_count / (float)class_i_count :  1 / (float)class_i_count;
				float transition_prob = (weights.interpolation[CLASS] == 0.0) ? 0.1 :  ngram_prob(&model_maps.class_map, i, class_i, class_i_count, model_metadata, sent_info.class_sent, sent_info.class_lengths, CLASSLEN, weights.class);
				the_class_prob = transition_prob * emission_prob;
				//printf("w=%s, w_i_cnt=%g, smooth=%g, class_i=%s, class_i_count=%i, prenorm_ngram_prob=%g, class_prob=%g, token_count=%lu, type_count=%u, line_count=%lu\n", word_i, (float)word_i_count, dklm_params.smooth, class_i, map_find_entry(&model_maps.class_map, class_i), the_ngram_prob, the_class_prob, model_metadata.token_count, model_metadata.type_count, model_metadata.line_count);
			}


			float score_i = the_class_prob;

			if (cmd_args.verbose > 0)
				printf("  class=%g, log10=%g, i=%i, class_i=%s\n", the_class_prob, log10(the_class_prob), i, class_i);


			sent_score += log2f(score_i); // Increment running sentence total

			if (cmd_args.verbose >= 0) {
				char word_string[KEYLEN];
				int word_len = sprintf(word_string, "%s=%u %f\t", word_i, word_i_count, log2f(score_i));
			}
#endif
		} // for i loop

		sum_log_probs += sent_score; // Increment running test set total, for perplexity
		free_sent_info(sent_info);
	} // Done querying current sentence
	return sum_log_probs;
}

void print_sent_info(struct_sent_info * restrict sent_info) {
	printf("struct sent_info { length = %u\n", sent_info->length);
	for (sentlen_t i = 0; i < sent_info->length; i++) {
		printf(" i=%u\twlen=%i\twcnt=%u\twcls=%u\tw=%s\n", i, sent_info->word_lengths[i], sent_info->sent_counts[i], sent_info->class_sent[i], sent_info->sent[i]);
	}
	printf("}\n");
}

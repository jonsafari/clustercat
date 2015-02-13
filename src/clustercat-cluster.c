#include "clustercat-cluster.h"
#include "clustercat-array.h"

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


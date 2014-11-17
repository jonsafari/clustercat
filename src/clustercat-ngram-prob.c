#include "clustercat.h"
#include "clustercat-data.h"
#include "clustercat-ngram-prob.h"
#include "clustercat-math.h"			// dot_product()


float ngram_prob(struct_map *ngram_map[const], const sentlen_t i, const char * restrict word_i, const unsigned int word_i_count, const struct_model_metadata model_metadata, char * restrict sent[const], const short word_lengths[const], const unsigned char ngram_order, const float weights[const]) { // Cf. increment_ngram()
	if (ngram_order == 0) // Do nothing
		return -1;

	const short start_position = (i >= ngram_order-1) ? i - (ngram_order-1) : 0; // N-grams starting point can be 0, for <s>
	sentlen_t history_len_used = i - start_position; // This value is subject to reduction; if a history is unattested, we must still backoff to give a probability distribution summing to one
	float order_probs[ngram_order];  // Unknown at 0, unigrams at 1, bigrams at 2, trigrams at 3, ...
	// model_metadata.token_count doesn't include </s> (nor <s>, but we don't consider <s>)
	float unigram_denominator = ((float)model_metadata.token_count + model_metadata.line_count);
	order_probs[0] = 1 / ((float)model_metadata.type_count - 1); // unknown prob.  We don't count <s>
	//printf("\nngram_prob0.5: i=%i, word_i=%s, ngram_order=%i, start_pos=%i, start_word=%s, hist_len_used=%i, unigram_denom=%g, unk_prob=%g, line=%i\n", i, sent[i], ngram_order, start_position, sent[start_position], history_len_used, unigram_denominator, order_probs[0], __LINE__); fflush(stdout);

	short j;
	unsigned long sizeof_char = sizeof(char); // We use this multiple times
	unsigned short strlen_word_i = strlen(word_i); // We use this multiple times
	unsigned char ngram_len = 0; // Terminating char '\0' is same size as joining tab, so we'll just count that later

	// We first build the longest n-gram string, then successively remove the leftmost word

	for (j = start_position; j < i ; ++j) { // Determine length of longest n-gram string
		ngram_len += sizeof_char + word_lengths[j]; // the additional sizeof_char is for either a space for words in the history, or for a \0 for word_i
		//printf("ngram_prob1: start_posn=%d, j=%d, w_j=%s, i=%i, w_i=%s, ngram_len=%d\n", start_position, j, sent[j], i, word_i, ngram_len);
	}
	ngram_len += sizeof_char + strlen_word_i; // Deal with word_i separately

	char ngram[ngram_len];
	if (start_position < i) { // Start multiword string
		strcpy(ngram, sent[start_position]);
		strcat(ngram, SECONDARY_SEP_STRING);
	} else { // Just a single word string
		strcpy(ngram, word_i);
		ngram[ngram_len] = '\0';
	}
	//printf("ngram_prob1.5: start_posn=%d, i=%i, w_i=%s, ngram_len=%d, ngram=<<%s>>\n", start_position, i, word_i, ngram_len, ngram);

	for (j = start_position+1; j < i ; ++j) { // Build longest n-gram string.  We do this as a separate loop than before since malloc'ing a bunch of times is probably more expensive than the first cheap loop
		strcat(ngram, sent[j]);
		strcat(ngram, SECONDARY_SEP_STRING);
	}

	if (start_position < i)
		strcat(ngram, word_i); // last word in n-gram string is manually specified (unless it's a unigram, which is already built)

	//printf("ngram_prob3: start_posn=%d, i=%i, w_i=%s, ngram_len=%d, ngram=<<%s>>\n", start_position, i, word_i, ngram_len, ngram);

	char * restrict jp = ngram; // Numerator
	short history_len = i - start_position;
	for (j = start_position; j <= i; ++j, --history_len) { // Traverse longest n-gram string.  TODO: traverse shortest string first && short-circuit if history not found

		if (history_len > 0) { // Bigrams and longer
			ngram[ngram_len - strlen_word_i - 2] = '\0'; // Allow us to get ngram history, but replacing the last space with a \0
			unsigned int denominator_count = map_find_entry(ngram_map, jp);
			//printf("ngram_prob3.7: history=<<%s>>, den._count=%u, token_count=%lu\n", jp, denominator_count, model_metadata.token_count);
			ngram[ngram_len - strlen_word_i - 2] = SECONDARY_SEP_CHAR; // Now restore it back to a space

			if (denominator_count) { // Denominator is greater than zero; history is found in training set
				float numerator_count = (float)map_find_entry(ngram_map, jp);
				order_probs[history_len+1] = numerator_count  / ((float)denominator_count);
				//printf("ngram_prob3.79: denominator is not zero :-)  num._count=%g, den._count=%u, history_len=%i, hist_len_used=%i, jp=%s\n", numerator_count, denominator_count, history_len, history_len_used, jp);
			} else { // Avoid 0/0 == -nan;  history is not in training set
				history_len_used--;
				//printf("ngram_prob3.8: denominator is zero :-(  den._count=%u, history_len+1=%i, hist_len_used=%i, jp=%s\n", denominator_count, history_len+1, history_len_used, jp);
				order_probs[history_len+1] = 0.0;
			}
		} else { // Unigram
			order_probs[history_len+1] = word_i_count / unigram_denominator;
			//printf("ngram_prob3.10: unigram: order_probs[%i]=%g,  word_i_count=%u, token_count=%lu\n", history_len+1, order_probs[history_len+1], word_i_count, model_metadata.token_count);
		}

		//printf("ngram_prob4: start_posn=%d, i=%i, w_i=%s, ngram_len=%d, history_len=%i, hist_len_used=%i, jp=<<%s>>, prob=%g, weight=%g\n", start_position, i, word_i, ngram_len, history_len, history_len_used, jp, order_probs[history_len+1], weights[history_len+1]);
		jp += sizeof_char + word_lengths[j];
	}

	// history_len_used+1 represents the last position in the weights array to use;
	// We always use the unknown weight [0].  For example, if we're looking at a unigram,
	// then it didn't use any history (=0), and it should use the unk weight [0] plus the unigram weight [1].

	//float the_dot_product =  dot_product(order_probs, weights, history_len_used+2);
	//fprint_array(stdout, order_probs, history_len_used+2, ", "); printf("\n");
	//fprint_array(stdout, weights, history_len_used+2, ", "); printf("\n");
	//printf("ngram_prob5: ngram_prob=%g, hist_len_used+1=%i\n", the_dot_product, history_len_used+1);
	//return the_dot_product;

	return dot_productf(order_probs, weights, history_len_used+2);
}

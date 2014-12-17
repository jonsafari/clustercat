#include "clustercat-map.h"

inline void map_add_entry(struct_map **map, char * restrict entry_key, unsigned int count) { // Based on uthash's docs
	struct_map *local_s;

	//HASH_FIND_STR(*map, entry_key, local_s);	// id already in the hash?
	//if (local_s == NULL) {
		local_s = (struct_map *)malloc(sizeof(struct_map));
		unsigned short strlen_entry_key = strlen(entry_key);
		local_s->key = malloc(strlen_entry_key + 1);
		strcpy(local_s->key, entry_key);
		HASH_ADD_KEYPTR(hh, *map, local_s->key, strlen_entry_key, local_s);
	//}
	local_s->count = count;
}

inline void map_add_class(struct_map_word_class **map, const char * restrict entry_key, const unsigned short entry_class) {
	struct_map_word_class *local_s;

	//HASH_FIND_STR(*map, entry_key, local_s);	// id already in the hash?
	//if (local_s == NULL) {
		local_s = (struct_map_word_class *)malloc(sizeof(struct_map_word_class));
		strncpy(local_s->key, entry_key, KEYLEN-1);
		HASH_ADD_STR(*map, key, local_s);
	//}
	local_s->class = entry_class;
}

inline void map_update_class(struct_map_word_class **map, const char * restrict entry_key, const unsigned short entry_class) {
	struct_map_word_class *local_s;

	HASH_FIND_STR(*map, entry_key, local_s);	// id already in the hash?
	if (local_s == NULL) {
		local_s = (struct_map_word_class *)malloc(sizeof(struct_map_word_class));
		strncpy(local_s->key, entry_key, KEYLEN-1);
		HASH_ADD_STR(*map, key, local_s);
	}
	local_s->class = entry_class;
}

inline unsigned int map_increment_entry(struct_map **map, const char * restrict entry_key) { // Based on uthash's docs
	struct_map *local_s;

	#pragma omp critical
	{
		HASH_FIND_STR(*map, entry_key, local_s);	// id already in the hash?
		if (local_s == NULL) {
			local_s = (struct_map *)malloc(sizeof(struct_map));
			local_s->count = 0;
			unsigned short strlen_entry_key = strlen(entry_key);
			local_s->key = malloc(strlen_entry_key + 1);
			strcpy(local_s->key, entry_key);
			HASH_ADD_KEYPTR(hh, *map, local_s->key, strlen_entry_key, local_s);
		}
	}
	#pragma omp atomic
	++local_s->count;
	return local_s->count;
}

inline unsigned int map_increment_entry_fixed_width(struct_map_class **map, const wclass_t entry_key[const]) { // Based on uthash's docs
	struct_map_class *local_s;
	size_t sizeof_key = sizeof(wclass_t) * CLASSLEN;
	//printf("map++: sizeof_key=%zu, CLASSLEN=%u, cls_entry=[%hu,%hu,%hu,%hu]\n", sizeof_key, CLASSLEN, entry_key[0], entry_key[1], entry_key[2], entry_key[3]);

	//#pragma omp critical // not needed since each thread gets its own class_map
	{
		//printf("***41***: sizeof_key=%lu, sizeof(wclass_t)=%lu, CLASSLEN=%u, key=<%u,%u,%u,%u>\n", sizeof_key, sizeof(wclass_t), CLASSLEN, entry_key[0], entry_key[1], entry_key[2], entry_key[3]); fflush(stdout);
		HASH_FIND(hh, *map, entry_key, sizeof_key, local_s); // id already in the hash?
		if (local_s == NULL) {
			local_s = (struct_map_class *)malloc(sizeof(struct_map_class));
			local_s->count = 0;
			memcpy(local_s->key, entry_key, sizeof_key);
			HASH_ADD(hh, *map, key, sizeof_key, local_s);
		}
		//printf("\t***42***: count: %u\n", local_s->count); fflush(stdout);
	}
	#pragma omp atomic
	++local_s->count;
	return local_s->count;
}

inline unsigned int map_find_entry_fixed_width(struct_map_class *map[const], const wclass_t entry_key[const]) { // Based on uthash's docs
	struct_map_class *local_s;
	size_t sizeof_key = sizeof(wclass_t) * CLASSLEN;
	unsigned int local_count = 0;
	//printf("map: cls_entry=[%hu,%hu,%hu]\n", entry_key[0], entry_key[1], entry_key[2]);

	HASH_FIND(hh, *map, entry_key, sizeof_key, local_s); // id already in the hash?
	if (local_s != NULL) { // Deal with OOV
		local_count = local_s->count;
	}
	return local_count;
}

inline unsigned int map_update_entry(struct_map **map, const char * restrict entry_key, const unsigned int count) { // Based on uthash's docs
	struct_map *local_s;

	#pragma omp critical
	{
		HASH_FIND_STR(*map, entry_key, local_s);	// id already in the hash?
		if (local_s == NULL) {
			local_s = (struct_map *)malloc(sizeof(struct_map));
			local_s->count = count;
			unsigned short strlen_entry_key = strlen(entry_key);
			local_s->key = malloc(strlen_entry_key + 1);
			strcpy(local_s->key, entry_key);
			HASH_ADD_KEYPTR(hh, *map, local_s->key, strlen_entry_key, local_s);
		} else {
			local_s->count += count;
		}
	}
	return local_s->count;
}

inline unsigned int map_update_entry_float(struct_map_float **map, const char * restrict entry_key, const float frac_count) { // Based on uthash's docs
	struct_map_float *local_s;

	#pragma omp critical
	{
		HASH_FIND_STR(*map, entry_key, local_s);	// id already in the hash?
		if (local_s == NULL) {
			local_s = (struct_map_float *)malloc(sizeof(struct_map_float));
			unsigned short strlen_entry_key = strlen(entry_key);
			local_s->key = malloc(strlen_entry_key + 1);
			strcpy(local_s->key, entry_key);
			local_s->frac_count = frac_count;
			HASH_ADD_KEYPTR(hh, *map, local_s->key, strlen_entry_key, local_s);
		} else {
			local_s->frac_count += frac_count;
		}
	}
	return local_s->frac_count;
}

inline unsigned int map_find_entry(struct_map *map[const], const char * restrict entry_key) { // Based on uthash's docs
	struct_map *local_s;
	unsigned int local_count = 0;

	HASH_FIND_STR(*map, entry_key, local_s);	// local_s: output pointer
	if (local_s != NULL) { // Deal with OOV
		local_count = local_s->count;
	}
	return local_count;
}

inline float map_find_entry_float(struct_map_float *map[const], const char * restrict entry_key) { // Based on uthash's docs
	struct_map_float *local_s;
	float local_count = 0.0;

	HASH_FIND_STR(*map, entry_key, local_s);	// local_s: output pointer
	if (local_s != NULL) { // Deal with OOV
		local_count = local_s->frac_count;
	}
	return local_count;
}

inline unsigned short get_class(struct_map_word_class *map[const], const char * restrict entry_key, const unsigned short unk) {
	struct_map_word_class *local_s;

	HASH_FIND_STR(*map, entry_key, local_s);	// local_s: output pointer
	if (local_s != NULL) { // Word is found
		return local_s->class;
	} else { // Word is not found
		return unk;
	}
}

unsigned int get_keys(struct_map *map[const], char *keys[]) {
	struct_map *entry, *tmp;
	unsigned int number_of_keys = 0;

	HASH_ITER(hh, *map, entry, tmp) {
		// Build-up array of keys
		unsigned short wlen = strlen(entry->key);
		keys[number_of_keys] = (char *) malloc(wlen + 1);
		strcpy(keys[number_of_keys], entry->key);
		number_of_keys++;
	}
	return number_of_keys;
}

void delete_entry(struct_map **map, struct_map *entry) { // Based on uthash's docs
	HASH_DEL(*map, entry);	// entry: pointer to deletee
	free(entry->key); // key is a malloc'd string
	free(entry);
}

void delete_all(struct_map **map) {
	struct_map *current_entry, *tmp;

	HASH_ITER(hh, *map, current_entry, tmp) { // Based on uthash's docs
		HASH_DEL(*map, current_entry);	// delete it (map advances to next)
		free(current_entry);	// free it
	}
}

void delete_all_class(struct_map_class **map) {
	struct_map_class *current_entry, *tmp;

	HASH_ITER(hh, *map, current_entry, tmp) { // Based on uthash's docs
		HASH_DEL(*map, current_entry);	// delete it (map advances to next)
		free(current_entry);	// free it
	}
}

void print_words_and_classes(struct_map_word_class **map) {
	struct_map_word_class *s;
	for (s = *map; s != NULL; s = (struct_map_word_class *)(s->hh.next))
		printf("%s\t%hu\n", s->key, s->class);
}

int count_sort(struct_map *a, struct_map *b) { // Based on uthash's docs
	return (a->count - b->count);
}

void sort_by_count(struct_map **map) { // Based on uthash's docs
	HASH_SORT(*map, count_sort);
}

int key_sort(struct_map_word_class *a, struct_map_word_class *b) {
	return strcmp(a->key, b->key);
}

void sort_by_key(struct_map_word_class **map) {
	HASH_SORT(*map, key_sort);
}

int class_sort(struct_map_word_class *a, struct_map_word_class *b) { // Based on uthash's docs
	return (a->class - b->class);
}

void sort_by_class(struct_map_word_class **map) {
	HASH_SORT(*map, class_sort);
}

unsigned long map_count(struct_map *map[const]) {
	return HASH_COUNT(*map);
}

unsigned long map_print_entries(struct_map **map, const char * restrict prefix, const char sep_char, const unsigned int min_count) {
	struct_map *entry, *tmp;
	unsigned long number_of_entries = 0;

	HASH_ITER(hh, *map, entry, tmp) {
		if (entry->count >= min_count) {
			printf("%s%s%c%i\n", prefix, entry->key, sep_char, entry->count);
			number_of_entries++;
		}
	}
	return number_of_entries;
}

unsigned long map_print_entries_float(struct_map_float **map, const char * restrict prefix, const char sep_char, const unsigned int min_count) {
	struct_map_float *entry, *tmp;
	unsigned long number_of_entries = 0;

	HASH_ITER(hh, *map, entry, tmp) {
		if (entry->frac_count >= min_count) {
			printf("%s%s%c%g\n", prefix, entry->key, sep_char, entry->frac_count);
			number_of_entries++;
		}
	}
	return number_of_entries;
}


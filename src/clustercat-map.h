#ifndef INCLUDE_CLUSTERCAT_MAP_HEADER
#define INCLUDE_CLUSTERCAT_MAP_HEADER

#include <stdio.h>
#include "uthash.h"

#ifdef ATA_STORE_KHASH
 #include "khash.h"
 KHASH_MAP_INIT_STR(struct_khash_float, float);
#endif

// Defaults
#define KEYLEN 80
#define CLASSLEN 4 // Longest possible class ngram to store
typedef unsigned short wclass_t;  // Number of possible word classes
typedef unsigned int   word_id_t; // Number of possible word classes

typedef struct {
	char * restrict key;
	unsigned int count;
	word_id_t word_id;
	UT_hash_handle hh;	// makes this structure hashable
} struct_map_word;

typedef struct { // Maps a class to its count
	wclass_t key[CLASSLEN];
	unsigned int count;
	UT_hash_handle hh;	// makes this structure hashable
} struct_map_class;

typedef struct { // Maps a word to its class
	char key[KEYLEN];
	wclass_t class;
	UT_hash_handle hh;	// makes this structure hashable
} struct_map_word_class;


void map_add_entry(struct_map_word **map, char * restrict entry_key, unsigned int count);

void map_add_class(struct_map_word_class **map, const char * restrict entry_key, const wclass_t entry_class);

void map_update_class(struct_map_word_class **map, const char * restrict entry_key, const wclass_t entry_class);

unsigned int map_increment_entry(struct_map_word **map, const char * restrict entry_key);

unsigned int map_increment_entry_fixed_width(struct_map_class **map, const wclass_t entry_key[const]);

unsigned int map_update_entry(struct_map_word **map, const char * restrict entry_key, const unsigned int count);

unsigned int map_find_entry(struct_map_word *map[const], const char * restrict entry_key);
unsigned int map_find_entry_fixed_width(struct_map_class *map[const], const wclass_t entry_key[const]);

wclass_t get_class(struct_map_word_class *map[const], const char * restrict entry_key, const wclass_t unk);

unsigned int get_keys(struct_map_word *map[const], char *keys[]);

void sort_by_class(struct_map_word_class **map);
void sort_by_key(struct_map_word_class **map);

unsigned long map_count(struct_map_word *map[const]);

unsigned long map_print_entries(struct_map_word **map, const char * restrict prefix, const char sep_char, const unsigned int min_count);
void print_words_and_classes(word_id_t type_count, char **unique_words, wclass_t word2class[restrict]);

void delete_all_class(struct_map_class **map);
void delete_entry(struct_map_word **map, struct_map_word *entry);

#endif // INCLUDE_HEADER

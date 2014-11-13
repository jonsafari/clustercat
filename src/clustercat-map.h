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
#define CLASSLEN 8

typedef struct {
	char * restrict key;
	unsigned int count;
	UT_hash_handle hh;	// makes this structure hashable
} struct_map;

typedef struct {
	char * restrict key;
	float frac_count;
	UT_hash_handle hh;	// makes this structure hashable
} struct_map_float;

typedef struct {
	char key[KEYLEN];
	unsigned short class; // could be string, but requires lots of restructuring
	UT_hash_handle hh;	// makes this structure hashable
} struct_map_word_class;


void map_add_entry(struct_map **map, char * restrict entry_key, unsigned int count);

void map_add_class(struct_map_word_class **map, const char * restrict entry_key, const unsigned short entry_class);

void map_update_class(struct_map_word_class **map, const char * restrict entry_key, const unsigned short entry_class);

unsigned int map_increment_entry(struct_map **map, const char * restrict entry_key);

unsigned int map_update_entry(struct_map **map, const char * restrict entry_key, const unsigned int count);

unsigned int map_update_entry_float(struct_map_float **map, const char * restrict entry_key, const float frac_count);

unsigned int map_find_entry(struct_map *map[const], const char * restrict entry_key);
float  map_find_entry_float(struct_map_float *map[const], const char * restrict entry_key);

unsigned short get_class(struct_map_word_class *map[const], const char * restrict entry_key, const unsigned short unk);

unsigned int get_keys(struct_map *map[const], char *keys[]);

unsigned long map_count(struct_map *map[const]);

unsigned long map_print_entries(struct_map **map, const char * restrict prefix, const char sep_char, const unsigned int min_count);

unsigned long map_print_entries_float(struct_map_float **map, const char * restrict prefix, const char sep_char, const unsigned int min_count);
#endif // INCLUDE_HEADER

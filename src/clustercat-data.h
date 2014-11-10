#ifndef INCLUDE_DKLM_DATA_HEADER
#define INCLUDE_DKLM_DATA_HEADER

#include "dklm-map.h"
#include "dklm-tree.h"

// Thanks Dipstick
#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x))
//	SHOW_DEFINE(DATA_STRUCT_FLOAT_NAME); // for example

// Default to storing word-word entries in hash table using uthash
// You can change this by compiling with -DATA_STORE_TREE_LCRS or -DATA_STORE_TRIE
#ifdef ATA_STORE_TRIE_LCRS
 #define DATA_STRUCT_FLOAT_HUMAN_NAME "hybrid_trie_lcrs"
 #define DATA_STRUCT_FLOAT_NAME word_word_float_trie_lcrs
 #define DATA_STRUCT_FLOAT_ADDR 
 #define DATA_STRUCT_FLOAT_SIZE sizeof(struct_tree_lcrs_float)
 #define DATA_STRUCT_FLOAT_TYPE struct_trie_lcrs_float *
 #define DATA_STRUCT_FLOAT_TYPE_IN_STRUCT struct_trie_lcrs_float *
 #define DECLARE_DATA_STRUCT_FLOAT DATA_STRUCT_FLOAT_TYPE DATA_STRUCT_FLOAT_NAME = NULL;
 #define INIT_DATA_STRUCT_FLOAT DATA_STRUCT_FLOAT_NAME = trie_lcrs_new(); // This should only be used once, initially. Hence no args
 #define UPDATE_ENTRY_FLOAT(db,key,val) ( trie_lcrs_update_entry_float((db), (key), (val)))
 #define FIND_ENTRY_FLOAT(db,key) ( trie_lcrs_find_entry_float((db), (key)))
 #define PRINT_ENTRIES_FLOAT(db, prefix, sep_char, min_count) ( trie_lcrs_print_entries_float((db), (prefix), (sep_char), (min_count)))
#elif ATA_STORE_TREE_LCRS
 #define DATA_STRUCT_FLOAT_HUMAN_NAME "tree_lcrs"
 #define DATA_STRUCT_FLOAT_NAME word_word_float_tree_lcrs
 #define DATA_STRUCT_FLOAT_ADDR 
 #define DATA_STRUCT_FLOAT_SIZE sizeof(struct_tree_lcrs_float)
 #define DATA_STRUCT_FLOAT_TYPE struct_tree_lcrs_float *
 #define DATA_STRUCT_FLOAT_TYPE_IN_STRUCT struct_tree_lcrs_float *
 #define DECLARE_DATA_STRUCT_FLOAT DATA_STRUCT_FLOAT_TYPE DATA_STRUCT_FLOAT_NAME = NULL;
 #define INIT_DATA_STRUCT_FLOAT DATA_STRUCT_FLOAT_NAME = tree_lcrs_new('\0', 0.0, NULL);
 #define UPDATE_ENTRY_FLOAT(db,key,val) ( tree_lcrs_update_entry_float((db), (key), 0, (val)))
 #define FIND_ENTRY_FLOAT(db,key) ( tree_lcrs_find_entry_float((db), (key)))
 #define PRINT_ENTRIES_FLOAT(db, prefix, sep_char, min_count) ( tree_lcrs_print_entries_float((db), (prefix), (sep_char), (min_count)))
#elif defined ATA_STORE_TRIE
 #define DATA_STRUCT_FLOAT_HUMAN_NAME "trie"
 #define DATA_STRUCT_FLOAT_NAME word_word_float_trie
 #define DATA_STRUCT_FLOAT_ADDR 
 #define DATA_STRUCT_FLOAT_SIZE sizeof(struct_trie_float)
 #define DATA_STRUCT_FLOAT_TYPE struct_trie_float * restrict
 #define DATA_STRUCT_FLOAT_TYPE_IN_STRUCT struct_trie_float *
 #define DECLARE_DATA_STRUCT_FLOAT DATA_STRUCT_FLOAT_TYPE DATA_STRUCT_FLOAT_NAME = NULL;
 #define INIT_DATA_STRUCT_FLOAT DATA_STRUCT_FLOAT_NAME = trie_new('\0');
 #define UPDATE_ENTRY_FLOAT(db,key,val) ( trie_update_entry_float((db), (key), (val)))
 #define FIND_ENTRY_FLOAT(db,key) ( trie_find_entry_float((db), (key)))
 #define PRINT_ENTRIES_FLOAT(db, prefix, sep_char, min_count) ( trie_print_entries_float((db), (prefix), (sep_char), (min_count)))
#elif defined ATA_STORE_KHASH // https://github.com/attractivechaos/klib
 #define DATA_STRUCT_FLOAT_HUMAN_NAME "khash_map"
 #define DATA_STRUCT_FLOAT_NAME word_word_float_khash
 #define DATA_STRUCT_FLOAT_ADDR 
 #define DATA_STRUCT_FLOAT_TYPE kh_struct_khash_float_t
 #define DATA_STRUCT_FLOAT_TYPE_IN_STRUCT kh_struct_khash_float_t
 #define DATA_STRUCT_FLOAT_SIZE sizeof(kh_struct_khash_float_t)
 #define DECLARE_DATA_STRUCT_FLOAT KHASH_MAP_INIT_STR(DATA_STRUCT_FLOAT_TYPE, float);
 #define INIT_DATA_STRUCT_FLOAT khash_t(struct_khash_float) * DATA_STRUCT_FLOAT_NAME = kh_init(struct_khash_float);
 #define UPDATE_ENTRY_FLOAT(db,key,val) { \
	 int ret; \
	 khint_t k = kh_put(struct_khash_float, (&db), (key), &ret); \
	 if (!ret) kh_del(struct_khash_float, (&db), (k)); \
	 kh_value((&db), (k)) = (val); \
 }
 #define FIND_ENTRY_FLOAT(db,key) ( kh_get(struct_khash_float, (db), (key)))
 //#define PRINT_ENTRIES_FLOAT(db, prefix, sep_char, min_count) ({ \
 //    unsigned long number_of_entries = 0; \
 //    for (khint_t k = kh_begin(db); k != kh_end(db); ++k) \
 //   	if (kh_exist(db, k)) { \
 //   		printf("foobar\n"); \
 ////		printf("%s%s%c%i\n", prefix, entry->key, sep_char, entry->count);
 //   		number_of_entries++; \
 //   	} \
 //    return number_of_entries; \
 //})
 #define PRINT_ENTRIES_FLOAT(db, prefix, sep_char, min_count) (1)
#else // Default to UThash map
 #define DATA_STRUCT_FLOAT_HUMAN_NAME "uthash_map"
 #define DATA_STRUCT_FLOAT_NAME word_word_float_map
 #define DATA_STRUCT_FLOAT_ADDR &
 #define DATA_STRUCT_FLOAT_SIZE sizeof(struct_map_float)
 #define DATA_STRUCT_FLOAT_TYPE struct_map_float **
 #define DATA_STRUCT_FLOAT_TYPE_IN_STRUCT struct_map_float *
 #define DECLARE_DATA_STRUCT_FLOAT struct_map_float *DATA_STRUCT_FLOAT_NAME  = NULL;	// Must initialize to NULL.
 #define INIT_DATA_STRUCT_FLOAT  // Don't need to do anything for uthash maps
 #define UPDATE_ENTRY_FLOAT(db,key,val) ( map_update_entry_float((db), (key), (val)))
 #define FIND_ENTRY_FLOAT(db,key) ( map_find_entry_float((db), (key)))
 #define PRINT_ENTRIES_FLOAT(db, prefix, sep_char, min_count) ( map_print_entries_float((db), (prefix), (sep_char), (min_count)))
#endif

typedef struct {
	struct_map *word_map;
	struct_map *word_word_map;
	DATA_STRUCT_FLOAT_TYPE_IN_STRUCT DATA_STRUCT_FLOAT_NAME;
	struct_map *ngram_map;
	struct_map *class_map;
	char **unique_words;
} struct_model_maps;


#endif // INCLUDE_HEADER

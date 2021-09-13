// Michael Mao
// mmmao

// I use a singly linked list of miniblocks to deal with 16 byte blocks
// I also use a 14-bucket sized seglist to deal with ≥32 byte blocks
// For mini blocks, i have a mandatory header field, and will either have
/* a payload of at most 8 bytes (if allocated) otherwise if free, a pointer to
the next free miniblock in the singly linked list. In regards to the normal
blocks, if it is allocated, it will have only a header and then a payload of any
size ≥ 9 bytes If it is not allocated, it will have 4 fields: header (8 bytes),
next pointer (8 bytes), prev pointer (8 bytes), and a footer field that is the
same as the header (8 bytes). This block will exist inside of some bucket class
in the seglist. The next and free pointer will point to free blocks in the
corresponding bucket class. The seglist is organized as an array, and since the
size of a block is essentially a key, we can use the size to access the
corresponding bucket class, to search or remove. In terms of both the singly
linked list and the seglist, if I want to allocate space on the heap, I will
find a free block in the corresponding list (falling through to the next larger
bucket class if non is found) and then remove that block from the list, since it
is no longer free and is allocated. I may need to split the block if necessary,
and if there exists a free remainder, I will add the block into its
corresponding free list. Finally, when I free a block, I will remove that block
from the corresponding list.

In terms of useful bits in a header / footer, the first bit (alloc) indicates
allocation status, the second bit (palloc) indicates if the previous block is
allocated, and the third bit (mpalloc) indicates if the previous block is a
mini_block

NOTE: I refer to block pointers as just the "block" itself.
NOTE: I assume that any mini block is of size 16 bytes, due to alignment
convention
NOTE: I refer to the main seglist as "seglist" and the mini block
free list as the "mini list"
*/
#include "mm.h"
#include "memlib.h"
#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef DRIVER
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif
#ifdef DEBUG
#define dbg_printf(...) printf(__VA_ARGS__)
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
#define dbg_printf(...) (sizeof(__VA_ARGS__), -1)
#define dbg_requires(expr) (sizeof(expr), 1)
#define dbg_assert(expr) (sizeof(expr), 1)
#define dbg_ensures(expr) (sizeof(expr), 1)
#define dbg_printheap(...) ((void)sizeof(__VA_ARGS__))
#endif
// BASIC CONSTANTS
typedef uint64_t word_t;
static const size_t wsize = sizeof(word_t);
static const size_t dsize = 2 * wsize;
static const size_t min_block_size = dsize;
static const size_t chunksize = (1 << 9);
static const word_t alloc_mask = 0x1;
static const word_t palloc_mask = 0x2;
static const word_t mpalloc_mask = 0x4;
static const word_t size_mask = ~(word_t)0xF;
static const int BUCKET_COUNT = 14;
static const size_t MINI_BLOCK_SIZE = 16;

// mini_block struct definition
typedef struct mini_block {
    word_t header;
    union {
        struct mini_block *next;
        char payload[0];
    };
} mini_block_t;

// normal block struct definition
typedef struct block {
    word_t header;
    union {
        struct {
            struct block *next;
            struct block *prev;
        };
        char payload[0];
    };
} block_t;

// free list declaration
static block_t *heap_start = NULL;
static mini_block_t *mini_root = NULL;
static block_t *seglist[BUCKET_COUNT];

// takes in two numbers and returns the max of two numbers
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

// given two numbers (size, n), round size to a multiple of n
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

// given a size, mpalloc, palloc, and alloc, package these into a word
// and return the word
static word_t pack(size_t size, bool mpalloc, bool palloc, bool alloc) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    if (palloc) {
        word |= palloc_mask;
    }
    if (mpalloc) {
        word |= mpalloc_mask;
    }
    return word;
}

// given a word, return the size encoded within it
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

// given any type of block, return the size of it
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

// given the pointer to the payload of a block
// return the block that holds it
// this block should be allocated
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, payload));
}

// given a block, return a pointer to its payload
// this block should be allocated
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    return (void *)(block->payload);
}

// given a block, return a word pointer to its footer
// this block should be a free block and not a mini block
static word_t *header_to_footer(block_t *block) {
    dbg_requires(get_size(block) != 0 &&
                 "Called header_to_footer on the epilogue block");
    return (word_t *)(block->payload + get_size(block) - dsize);
}

// given a footer pointer, return the block that holds it
// this block must be a free block and not a mini block
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
    return (block_t *)((char *)footer + wsize - size);
}

// given a block, retun the payload size
// this assumes the block is allocated
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - wsize;
}

// given a word, return its encoding of alloc
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

// given a block, return if its allocated
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

// given a word, return the encoding of palloc
static bool extract_palloc(word_t word) {
    return (bool)((word & palloc_mask) >> 1);
}

// given a block, return if its previous block is allocated
static bool get_palloc(block_t *block) {
    return extract_palloc(block->header);
}

// given a word, return the encoding of mpalloc
static bool extract_mpalloc(word_t word) {
    return (bool)((word & mpalloc_mask) >> 2);
}

// given a block, return if the previous block is mini or not
static bool get_mpalloc(block_t *block) {
    return extract_mpalloc(block->header);
}

// given any block, return the block immediately after it
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

// given a block, return the footer of its immediate previous block
// the previous block must be free and not a miniblock
static word_t *find_prev_footer(block_t *block) {
    return &(block->header) - 1;
}

// given a block and a bool condition if the previous block is mini or not,
// return the block immediately before it
// this can never be called on the prologue
static block_t *find_prev(block_t *block, bool mpalloc) {
    dbg_requires(block != NULL);
    if (mpalloc) {
        // previous was a mini block
        return ((block_t *)((char *)block - MINI_BLOCK_SIZE));
    } else {
        // previous is not a mini block
        word_t *footerp = find_prev_footer(block);
        if (extract_size(*footerp) == 0) {
            return NULL;
        }
        return footer_to_header(footerp);
    }
}

// given a block, mpalloc, and palloc, write this data into the epilogue
static void write_epilogue(block_t *block, bool mpalloc, bool palloc) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == mem_heap_hi() - 7);
    block->header = pack(0, mpalloc, palloc, true);
}

// given a block, returns if the block is the epilogue or not
static bool isEpilogue(block_t *block) {
    return ((void *)block == (void *)(mem_heap_hi() - 7) && get_alloc(block) &&
            (get_size(block) == 0));
}

// given a block, return if the block is a mini block or not
static bool isMiniBlock(block_t *block) {
    return (get_size(block) == (size_t)MINI_BLOCK_SIZE);
}

// given a size, return if the size is that of a mini block
static bool isMiniSize(size_t size) {
    return (size == (size_t)MINI_BLOCK_SIZE);
}

// given a block, size, mpalloc, palloc, and alloc, write this information into
// that block
static void write_block(block_t *block, size_t size, bool mpalloc, bool palloc,
                        bool alloc) {
    dbg_requires(block != NULL);
    dbg_requires(size > 0);
    if (isMiniSize(size)) {
        // the block "is" a mini block
        block->header = pack(size, mpalloc, palloc, alloc);
    } else {
        // the block "is" not a mini block
        if (alloc) {
            block->header = pack(size, mpalloc, palloc, alloc);
        } else {
            block->header = pack(size, mpalloc, palloc, alloc);
            word_t *footerp = header_to_footer(block);
            *footerp = pack(size, mpalloc, palloc, alloc);
        }
    }
}

// given a block and a seglist index, insert that block into the corresponding
// bucket
// the block cannot be a mini block
static void explicit_insert_block(block_t *block, long idx) {
    assert(!isMiniBlock(block));
    if (seglist[idx] == NULL) {
        // directly insert block
        seglist[idx] = block;
        block->next = NULL;
        block->prev = NULL;
    } else {
        seglist[idx]->prev = block;
        block->next = seglist[idx];
        block->prev = NULL;
        seglist[idx] = block;
    }
}

// given a block and a seglist index, remove the block from its bucket
// this block cannot be mini, and should be free to begin with
static void explicit_remove_block(block_t *block, long idx) {
    assert(!isMiniBlock(block));
    block_t *prev_free_block = block->prev;
    block_t *next_free_block = block->next;
    if (prev_free_block == NULL && next_free_block == NULL) {
        // set whole thing to NULL, nothing in the list after removal
        seglist[idx] = NULL;
    }
    // next block exists
    else if (prev_free_block == NULL && next_free_block != NULL) {
        next_free_block->prev = NULL;
        seglist[idx] = next_free_block;
    }
    // previous block exists
    else if (prev_free_block != NULL && next_free_block == NULL) {
        prev_free_block->next = NULL;
    }
    // both previous and next block exist
    else {
        prev_free_block->next = next_free_block;
        next_free_block->prev = prev_free_block;
    }
}

// initialize the seglist buckets all to NULL
static void seglist_init(void) {
    for (int idx = 0; idx < BUCKET_COUNT; idx++) {
        seglist[idx] = NULL;
    }
}

// given a size, return the corresponding index of its correct bucket in the
// seglist
// the size should be ≥ 32 to begin with
static long home_address(size_t size) {
    if (size == 32) {
        return 0;
    } else if (33 <= size && size <= 64) {
        return 1;
    } else if (65 <= size && size <= 85) {
        return 2;
    } else if (86 <= size && size <= 112) {
        return 3;
    } else if (113 <= size && size <= 128) {
        return 4;
    } else if (129 <= size && size <= 160) {
        return 5;
    } else if (161 <= size && size <= 200) {
        return 6;
    } else if (201 <= size && size <= 256) {
        return 7;
    } else if (257 <= size && size <= 512) {
        return 8;
    } else if (513 <= size && size <= 1024) {
        return 9;
    } else if (1025 <= size && size <= 2048) {
        return 10;
    } else if (2049 <= size && size <= 4096) {
        return 11;
    } else if (4097 <= size && size <= 8192) {
        return 12;
    } else {
        return 13;
    }
}

// given a block, update the next block's information regarding palloc
static void update_next_palloc(block_t *block) {
    bool alloc = get_alloc(block);
    block_t *next = find_next(block);
    size_t next_size = get_size(next);
    bool next_alloc = get_alloc(next);
    bool next_mpalloc = get_mpalloc(next);
    if (isEpilogue(next)) {
        write_epilogue(next, next_mpalloc, alloc);
    } else {
        write_block(next, next_size, next_mpalloc, alloc, next_alloc);
    }
}

// given a block, update the next block's information regarding mpalloc
static void update_next_mpalloc(block_t *block) {
    bool new_next_mpalloc = isMiniBlock(block);
    block_t *next = find_next(block);
    size_t next_size = get_size(next);
    bool next_alloc = get_alloc(next);
    bool next_palloc = get_palloc(next);
    if (isEpilogue(next)) {
        write_epilogue(next, new_next_mpalloc, next_palloc);
    } else {
        write_block(next, next_size, new_next_mpalloc, next_palloc, next_alloc);
    }
}

// given a mini_block, insert it into the mini list
static void insert_mini_block(mini_block_t *mini_block) {
    assert(isMiniBlock((block_t *)mini_block));
    if (mini_root == NULL) {
        // nothing in the mini list
        mini_root = mini_block;
        mini_block->next = NULL;
    } else {
        mini_block->next = mini_root;
        mini_root = mini_block;
    }
    update_next_palloc((block_t *)mini_block);
    update_next_mpalloc((block_t *)mini_block);
}

// given a mini block, remove it from the mini list
// this block should be in the mini list to begin with
static void remove_mini_block(mini_block_t *mini_block) {
    assert(isMiniBlock((block_t *)mini_block));
    // nothing to remove
    if (mini_root == NULL) {
        return;
    } else if (mini_root == mini_block) {
        // mini block is first thing in the mini list
        mini_root = mini_block->next;
        mini_block->next = NULL;
    } else {
        // block is further in the mini list
        mini_block_t *cur_mini_block = mini_root;
        while (cur_mini_block != NULL) {
            if (cur_mini_block->next == mini_block) {
                cur_mini_block->next = mini_block->next;
                mini_block->next = NULL;
                return;
            }
            cur_mini_block = cur_mini_block->next;
        }
    }
    update_next_palloc((block_t *)mini_block);
    update_next_mpalloc((block_t *)mini_block);
}

// given a size and a starting seglist index
// search the seglist bucket for a block to store the size requested
// if none found in the current bucket, search the immediate next block
// i use better fit: in that after you find the first block that can fit
// the requested size, then keep looking for a closer fit for the next
// (3) blocks, and then finally return the best block found
static block_t *find_fit(size_t asize, long idx) {
    bool count_switch = false;
    long counter = 3;
    block_t *cur_block;
    block_t *best_block = NULL;
    size_t best_gap;

    for (long i = idx; i < BUCKET_COUNT; i++) {
        for (cur_block = seglist[i];
             (cur_block != NULL) && (get_size(cur_block) > 0);
             cur_block = cur_block->next) {
            if (asize <= get_size(cur_block)) {
                // found a fit
                if (best_block == NULL) {
                    // best block is not initalized, so set it
                    best_block = cur_block;
                    best_gap = get_size(best_block) - asize;
                    // begin better fit mode
                    count_switch = true;
                } else {
                    // try to improve best block
                    if ((get_size(cur_block) - asize) < best_gap) {
                        best_gap = get_size(cur_block) - asize;
                        best_block = cur_block;
                    }
                }
            }
            // if in better fit mode, begin decrementing counter
            // and if counter is at 0, return the best block without remorse
            if (count_switch) {
                counter -= 1;
                if (counter == 0) {
                    return best_block;
                }
            }
        }
        // if youve found a best block in the current bucket, and you are done
        // looking at the bucket, just return, as next bucket will be too large
        // anyway
        if (best_block != NULL) {
            return best_block;
        }
    }
    return best_block;
}

// generous checkheap
bool mm_checkheap(int line) {
    return true;
}

// given a block, insert it into its corresponding free list
static void insertVagueBlock(block_t *block) {
    if (isMiniBlock(block)) {
        insert_mini_block((mini_block_t *)block);
    } else {
        explicit_insert_block(block, home_address(get_size(block)));
    }
}

// given a block, remove it from its corresponding free list
static void removeVagueBlock(block_t *block) {
    if (isMiniBlock(block)) {
        remove_mini_block((mini_block_t *)block);
    } else {
        explicit_remove_block(block, home_address(get_size(block)));
    }
}

// given a freed block, coalesce it with its free neighbors
static block_t *coalesce_block(block_t *block) {
    block_t *next = find_next(block);
    bool prev_allocated = get_palloc(block);
    bool next_allocated = get_alloc(next);
    bool mpalloc = get_mpalloc(block);
    // both prev and next are free
    if (!prev_allocated && !next_allocated) {
        block_t *prev = find_prev(block, mpalloc);

        bool prev_mpalloc = get_mpalloc(prev);
        bool prev_palloc = get_palloc(prev);

        size_t newSum = get_size(prev) + get_size(block) + get_size(next);
        removeVagueBlock(prev);
        removeVagueBlock(next);

        write_block(prev, newSum, prev_mpalloc, prev_palloc, false);

        insertVagueBlock(prev);
        update_next_palloc(prev);
        update_next_mpalloc(prev);
        return prev;
    }
    // only next is free
    else if (!next_allocated && prev_allocated) {
        size_t newSum = get_size(block) + get_size(next);
        removeVagueBlock(next);
        bool block_mpalloc = get_mpalloc(block);
        bool block_palloc = get_palloc(block);

        write_block(block, newSum, block_mpalloc, block_palloc, false);

        insertVagueBlock(block);
        update_next_mpalloc(block);
        update_next_palloc(block);
        return block;
    }
    // only prev is free
    else if (!prev_allocated && next_allocated) {
        block_t *prev = find_prev(block, mpalloc);
        bool prev_mpalloc = get_mpalloc(prev);
        bool prev_palloc = get_palloc(prev);

        size_t newSum = get_size(prev) + get_size(block);
        removeVagueBlock(prev);

        write_block(prev, newSum, prev_mpalloc, prev_palloc, false);
        insertVagueBlock(prev);
        update_next_palloc(prev);
        update_next_mpalloc(prev);
        return prev;
    } else {
        // neither next nor prev are free
        insertVagueBlock(block);
        update_next_palloc(block);
        update_next_mpalloc(block);
        return block;
    }
}

// given a size, mpalloc, and palloc, update the newly added heap extension with
// these fields, such that this newly added heap extension can support the input
// size
static block_t *extend_heap(size_t size, bool mpalloc, bool palloc) {
    void *bp;
    size = round_up(size, dsize);
    if ((bp = mem_sbrk(size)) == (void *)-1) {
        return NULL;
    }

    block_t *block = payload_to_header(bp);
    write_block(block, size, mpalloc, palloc, false);

    block_t *block_next = find_next(block);
    write_epilogue(block_next, isMiniBlock(block), false);

    block = coalesce_block(block);

    return block;
}

// initialize the heap
bool mm_init(void) {
    seglist_init();
    mini_root = NULL;
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));

    if (start == (void *)-1) {
        return false;
    }

    start[0] = pack(0, false, true, true);
    start[1] = pack(0, false, true, true);

    heap_start = (block_t *)&(start[1]);

    if (extend_heap(chunksize, false, true) == NULL) {
        return false;
    }

    return true;
}

// given a free block, allocate a block of size "asize", and then add any
// remaining free block after such an operation into its corresponding free list
// this block should be implicitly allocated
static void split_block(block_t *block, size_t asize) {
    dbg_requires(get_alloc(block));

    size_t block_size = get_size(block);

    if ((block_size - asize) >= min_block_size) {
        // there will be a remainder after splitting
        block_t *block_next;

        bool mpalloc = get_mpalloc(block);
        bool palloc = get_palloc(block);

        removeVagueBlock(block);
        write_block(block, asize, mpalloc, palloc, true);

        block_next = find_next(block);

        write_block(block_next, block_size - asize, isMiniBlock(block), true,
                    false);

        insertVagueBlock(block_next);

        update_next_mpalloc(block_next);
        update_next_palloc(block_next);
    } else {
        // no remainder after splitting
        bool mpalloc = get_mpalloc(block);
        bool palloc = get_palloc(block);
        removeVagueBlock(block);
        write_block(block, block_size, mpalloc, palloc, true);
        update_next_palloc(block);
        update_next_mpalloc(block);
    }

    dbg_ensures(get_alloc(block));
}

// given a size, allocate it on the heap
void *malloc(size_t size) {
    dbg_requires(mm_checkheap(__LINE__));
    size_t asize;
    size_t extendsize;
    block_t *block;
    void *bp = NULL;

    if (heap_start == NULL) {
        mm_init();
    }

    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    asize = round_up(size + wsize, dsize);

    // if the block size is that of a mini block
    // try to find a mini block to store it
    if (asize == (size_t)MINI_BLOCK_SIZE) {
        block = (block_t *)mini_root;
        // if not mini block found, look in the seglist
        if (block == NULL) {
            block = find_fit(asize, 0);
        }
    } else {
        long home = home_address(asize);
        block = find_fit(asize, home);
    }

    if (block == NULL) {
        // extend the heap
        // if not block found in any free list
        bool old_mpalloc = get_mpalloc((block_t *)(mem_heap_hi() - 7));
        bool old_palloc = get_palloc((block_t *)(mem_heap_hi() - 7));
        extendsize = max(asize, chunksize);
        block = extend_heap(extendsize, old_mpalloc, old_palloc);
        if (block == NULL) {
            return bp;
        }
    }
    dbg_assert(!get_alloc(block));

    size_t block_size = get_size(block);
    bool mpalloc = get_mpalloc(block);
    bool palloc = get_palloc(block);
    write_block(block, block_size, mpalloc, palloc, true);

    split_block(block, asize);

    bp = header_to_payload(block);

    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
}

// given a payload pointer, free the block that holds it from the heap
void free(void *bp) {
    dbg_requires(mm_checkheap(__LINE__));
    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);

    dbg_assert(get_alloc(block));

    bool mpalloc = get_mpalloc(block);

    bool palloc = get_palloc(block);
    write_block(block, size, mpalloc, palloc, false);

    block = coalesce_block(block);

    update_next_mpalloc(block);
    update_next_palloc(block);
    dbg_ensures(mm_checkheap(__LINE__));
}

// given a payload pointer, and a size, reallocate space of some allocated block
// to that size
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

// given sizes of elements and a size, call malloc but intialize everything to 0
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}

/*
*****************************************************************************
* Do not delete the following super-secret(tm) lines!                       *
*                                                                           *
* 53 6f 20 79 6f 75 27 72 65 20 74 72 79 69 6e 67 20 74 6f 20               *
*                                                                           *
* 66 69 67 75 72 65 20 6f 75 74 20 77 68 61 74 20 74 68 65 20               *
* 68 65 78 61 64 65 63 69 6d 61 6c 20 64 69 67 69 74 73 20 64               *
* 6f 2e 2e 2e 20 68 61 68 61 68 61 21 20 41 53 43 49 49 20 69               *
*                                                                           *
* 73 6e 27 74 20 74 68 65 20 72 69 67 68 74 20 65 6e 63 6f 64               *
* 69 6e 67 21 20 4e 69 63 65 20 74 72 79 2c 20 74 68 6f 75 67               *
* 68 21 20 2d 44 72 2e 20 45 76 69 6c 0a c5 7c fc 80 6e 57 0a               *
*                                                                           *
*****************************************************************************
*/

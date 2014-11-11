#!/usr/bin/env make

CC=cc
INCLUDE=-I ./src/ext/uthash/src/
##  * For -march info on your platform, type: gcc -march=native -Q --help=target  (or just compile with -march=native )
##  * We include the argument -Wno-unknown-pragmas to suppress clang's lack of support for openmp
##    Since we use the gnuism 'override', you don't need to modify this makefile; you can just run:  make -j4 CFLAGS=-DATA_STORE_TRIE_LCRS
override CFLAGS += -std=c99 -O3 -fopenmp -finline-functions -fno-math-errno -fstrict-aliasing -DHASH_FUNCTION=HASH_SFH -DHASH_BLOOM=25 -Wall -Wextra -Winline -Wstrict-aliasing -Wno-unknown-pragmas -Wno-unused-parameter -Wno-comment ${INCLUDE}
LDLIBS=-lm -lz #-ltcmalloc_minimal
BIN=bin/
SRC=src/
OBJS=${SRC}/clustercat-array.o ${SRC}/clustercat-io.o ${SRC}/clustercat-map.o ${SRC}/clustercat-tokenize.o
includes=${SRC}/$(wildcard *.h)
date:=$(shell date +%F)
machine_type:=$(shell uname -m)

all: ${BIN}/clustercat
.PHONY : all clean

clustercat.h: ${SRC}/clustercat-array.h ${SRC}/clustercat-data.h ${SRC}/clustercat-map.h


${BIN}/clustercat: ${SRC}/clustercat.c ${OBJS}
	${CC} $^ -o $@ ${CFLAGS} ${LDLIBS}

clustercat.c: ${SRC}/clustercat.h ${SRC}/clustercat-io.h ${SRC}/clustercat-tokenize.h

tar: ${BIN}/clustercat
	mkdir clustercat-${date} && \
	mkdir clustercat-${date}/bin && \
	mkdir clustercat-${date}/src && \
	mkdir --parents clustercat-${date}/src/ext/uthash/src && \
	cp -a ${BIN}/clustercat clustercat-${date}/bin/ && \
	cp -a ${BIN}/clustercat clustercat-${date}/bin/clustercat.${machine_type} && \
	cp -a ${SRC}/*.c ${SRC}/*.h clustercat-${date}/src/ && \
	cp -a Makefile README.md LICENSE clustercat-${date}/ && \
	cp -a ${SRC}/ext/uthash/src/uthash.h clustercat-${date}/src/ext/uthash/src/ && \
	tar -cf clustercat-${date}.tar clustercat-${date}/ && \
	gzip -9 clustercat-${date}.tar && \
	rm -rf clustercat-${date}/

clean:
	\rm -f ${BIN}/clustercat ${SRC}/*.o

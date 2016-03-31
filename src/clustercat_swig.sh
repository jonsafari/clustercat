#!/bin/sh
## Temporary file
set -x
swig3.0 -python clustercat.i

INCLUDE='-I ext/uthash/src/'
CFLAGS="-march=native -std=c99 -O3 -fopenmp -finline-functions -fno-math-errno -fstrict-aliasing -DHASH_FUNCTION=HASH_SAX -DHASH_BLOOM=25 -Wall -Wextra -Winline -Wstrict-aliasing -Wno-unknown-pragmas -Wno-comment -Wno-missing-field-initializers ${INCLUDE}"
LDLIBS='-lm'
SRC=.
OBJS="${SRC}/clustercat-array.o ${SRC}/clustercat-cluster.o ${SRC}/clustercat-dbg.o ${SRC}/clustercat-io.o ${SRC}/clustercat-import-class-file.o ${SRC}/clustercat-map.o ${SRC}/clustercat-math.o ${SRC}/clustercat-tokenize.o"
clang -c -fpic   clustercat.c clustercat_wrap.c  ${CFLAGS}  -I /usr/include/python2.7/ ${LDLIBS}

ld -shared ${OBJS} clustercat.o clustercat_wrap.o -o _clustercat.so

echo 'import clustercat' | python

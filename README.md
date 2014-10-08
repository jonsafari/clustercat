
# ClusterCat: Flexible, Multithreaded Word-Class Induction Software


## Overview

ClusterCat induces word classes from unannotated text.
It is dual-licensed under the [LGPL v3][lgpl3] or the [Apache License v2][al2].
It is programmed in modern C, with few external libraries.

## System Requirements
- A **Unix**-like system (eg. Linux, FreeBSD, Mac OS X, Cygwin) that includes `make`.
- A modern **C** compiler that supports [C99][], and preferably [OpenMP][] v3.1+ .
  The OpenMP pragmas (for multi-threaded use) may be ignored by older compilers, and will not affect the output of the programs.
  - GCC 4.7+
  - Clang 3.0+ (current stable versions simply ignore OpenMP pragmas, which is ok)
  - Probably other modern C compilers that fully support C99 (not MSVC)
- That's it!

## Compilation
      make -j 4

## Commands
The binary `clustercat` gets compiled into the `bin` directory.
Command-line argument usage may be obtained by running with program with the **`--help`** flag.

**Training** preprocessed text (already tokenized, normalized, etc) is pretty simple:

      clustercat [options] < train.tok.txt > output.txt

The word-classes can be induced from [Brown clustering][] or the [Exchange Algorithm][], for example.
The format of the class file has each line consisting of `word`*TAB*`class` (a word type, then tab, then class).


## Citation
...

[lgpl3]: https://www.gnu.org/copyleft/lesser.html
[al2]: https://www.apache.org/licenses/LICENSE-2.0.html
[c99]: https://en.wikipedia.org/wiki/C99
[openmp]: https://en.wikipedia.org/wiki/OpenMP
[brown clustering]: https://en.wikipedia.org/wiki/Brown_clustering
[exchange algorithm]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.2354

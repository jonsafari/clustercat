
# ClusterCat: Fast, Flexible, Fun Word Class Clustering Software


## Overview

ClusterCat induces word classes from unannotated text.
It is freely licensed under the [LGPL v3][lgpl3] or the [MPL v2][mpl2].
It is programmed in modern C, with few external libraries.

## System Requirements
- A **Unix**-like system (eg. Linux, FreeBSD, Mac OS X, Cygwin) that includes `make`.
- A modern **C** compiler that supports [C99][], and preferably [OpenMP][] v3.1+ .
  The OpenMP pragmas (for multi-threaded use) may be ignored by older compilers, and will not affect the output of the programs.
  - GCC 4.7+
  - Clang 3.0+  current stable versions simply ignore OpenMP pragmas, which is ok. The program will give the same output, but without multithreaded support it will take longer.
  - Probably other modern C compilers that fully support C99 (not MSVC)
- That's it!

## Compilation
      make -j 4

## Commands
The binary program `clustercat` gets compiled into the `bin` directory.
Command-line argument usage may be obtained by running with program with the **`--help`** flag.

**Clustering** preprocessed text (already tokenized, normalized, etc) is pretty simple:

      clustercat [options] < train.tok.txt > output.txt

The word-classes are induced from a bidirectional [predictive][] [exchange algorithm][].
Future work includes support for [Brown clustering][].
The format of the class file has each line consisting of `word`*TAB*`class` (a word type, then tab, then class).

## Features
- Print **[word vectors][]** (a.k.a. word embeddings) using the `--word-vectors` flag.
- Start training using an **existing word cluster mapping** from other clustering software (eg. mkcls) using the `--class-file` flag.
- Adjust the number of **threads** to use with the `--jobs` flag.  The default is 4.
- Set the **minimum count** of words to consider using the `--min-count` flag.  The default is 2.
- Adjust the **number of clusters** or vector dimensions using the `--num-classes` flag. The default is the square root of the vocabulary size.

## Citation
...

[lgpl3]: https://www.gnu.org/copyleft/lesser.html
[mpl2]: https://www.mozilla.org/MPL/2.0
[c99]: https://en.wikipedia.org/wiki/C99
[openmp]: https://en.wikipedia.org/wiki/OpenMP
[predictive]: https://www.aclweb.org/anthology/P/P08/P08-1086.pdf
[exchange algorithm]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.2354
[brown clustering]: https://en.wikipedia.org/wiki/Brown_clustering
[word vectors]: https://en.wikipedia.org/wiki/Word_embedding

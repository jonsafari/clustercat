
# ClusterCat: Fast, Flexible, Fun Word Class Clustering Software


## Overview

ClusterCat induces word classes from unannotated text.
It is freely licensed under the [LGPL v3][lgpl3] or the [MPL v2][mpl2].
It is programmed in modern C, with few external libraries.

## System Requirements
- A **Unix**-like system (eg. Linux, FreeBSD, Mac OS X, Cygwin) that includes `make`.
- A modern **C** compiler that supports [C99][], and preferably [OpenMP][] v3.1+ .
  The OpenMP pragmas (for multi-threaded use) may be ignored by older compilers, and will not affect the output of the program.
  - GCC 4.7+ (recommended)
  - Clang 3.0+  current stable versions simply ignore OpenMP pragmas, which is ok. The program will give the same output, but without multithreaded support it will take longer.
  - Probably other modern C compilers that fully support C99 (not MSVC)
- That's it!

## Compilation
      make -j 4

## Commands
The binary program `clustercat` gets compiled into the `bin` directory.

**Clustering** preprocessed text (already tokenized, normalized, etc) is pretty simple:

      bin/clustercat [options] < train.tok.txt > clusters.tsv

The word-classes are induced from a bidirectional [predictive][] [exchange algorithm][].
Future work includes support for [Brown clustering][].
The format of the class file has each line consisting of `word`*TAB*`class` (a word type, then tab, then class).

Command-line argument usage may be obtained by running with program with the **`--help`** flag:

      bin/clustercat --help

## Features
- Print **[word vectors][]** (a.k.a. word embeddings) using the `--word-vectors` flag.  The binary format is compatible with word2vec's tools.
- Start training using an **existing word cluster mapping** from other clustering software (eg. mkcls) using the `--class-file` flag.
- Adjust the number of **threads** to use with the `--jobs` flag.  The default is 4.
- Adjust the **number of clusters** or vector dimensions using the `--num-classes` flag. The default is proportional to the square root of the vocabulary size.
- ClusterCat prints regular updates of approximately how much time remains, and about **what time it will finish**.
- Includes **compatibility wrapper script ` bin/mkcls `** that can be run just like mkcls.  You can use more classes now :-)

## Visualization
See [bl.ocks.org][] for cool data visualization of the clusters for various languages, including English, German, Persian, Hindi, Czech, Catalan, Tajik, Basque, Russian, French, and Maltese.

You can generate your own graphics from ClusterCat's output.
Add the flag  `--print-freqs`  to ClusterCat, then type the command:

      bin/flat_clusters2json.pl --word-labels < clusters.tsv > visualization/d3/clusters.json

You can either upload the [JSON][] file to [gist.github.com][], following instructions on the [bl.ocks.org](http://bl.ocks.org) front page, or you can view the graphic locally by running a minimal webserver in the `visualization/d3` directory:

      python -m SimpleHTTPServer 8116 2>/dev/null &

Then open a tab in your browser to [localhost:8116](http://localhost:8116) .

The default settings are sensible for normal usage, but for visualization you probably want much fewer word types and clusters -- less than 10,000 word types and 120 clusters.
Your browser will thank you.

## Perplexity
The perplexity that ClusterCat reports uses a bidirectional trigram class language model, which is much richer than the simple unidirectional bigram-based perplexities reported by most other software.
Richer models provide a better evaluation of the quality of clusters, having more sensitivity (power) to detect improvements.
If you want to directly compare the quality of clusters with a different program's output, you have a few options:

1. Load another clustering using `--class-file` , and see what the other clustering's initial bidirectional trigram perplexity is before any words get exchanged.
2. Use an external class-based language model.
3. Evaluate on a downstream task.


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
[bl.ocks.org]: http://bl.ocks.org/jonsafari
[JSON]: https://en.wikipedia.org/wiki/JSON
[gist.github.com]: https://gist.github.com

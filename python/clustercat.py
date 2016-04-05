#!/usr/bin/env python3
# By Jon Dehdari, 2016
# MIT License
# Simple Python wrapper for ClusterCat

import sys
import os.path
import argparse
parser = argparse.ArgumentParser(description='Converts words to integers, online')

parser.add_argument('-i', '--in_map', help="Load pre-existing mapping file")
parser.add_argument('-o', '--out_map', help="Save final mapping to file")
args = parser.parse_args()


def cluster(text=None, infile=None, classes=None, class_file=None, class_offset=None, forward_lambda=None, ngram_input=None, min_count=None, out=None, print_freqs=None, quiet=None, refine=None, rev_alternate=None, threads=None, tune_cycles=None, unidirectional=None, verbose=None, word_vectors=None):
    clusters = {}
    args = "clustercat "

    if (infile):
        args += " --in " + str(infile)

    if (classes):
        args += " --classes " + str(classes)

    if (class_file):
        args += " --class-file " + str(class_file)

    if (class_offset):
        args += " --class-offset " + str(class_offset)

    if (forward_lambda):
        args += " --forward-lambda " + str(forward_lambda)

    if (ngram_input):
        args += " --ngram-input "

    if (min_count):
        args += " --min-count " + str(min_count)

    if (out):
        args += " --out " + out

    if (print_freqs):
        args += " --print-freqs "

    if (quiet):
        args += " --quiet "

    if (refine):
        args += " --refine " + str(refine)

    if (rev_alternate):
        args += " --rev-alternate " + str(rev_alternate)

    if (threads):
        args += " --threads " + str(threads)

    if (tune_cycles):
        args += " --tune-cycles " + str(tune_cycles)

    if (unidirectional):
        args += " --unidirectional "

    if (verbose):
        args += " --verbose "

    if (word_vectors):
        args += " --word-vectors " + word_vectors

    print(args, file=sys.stderr)
    return clusters

def main():
    cluster(infile=sys.stdin)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# By Jon Dehdari, 2016
# MIT License
# Simple Python wrapper for ClusterCat

import sys, argparse, subprocess
parser = argparse.ArgumentParser(description='Converts words to integers, online')

parser.add_argument('-i', '--in_map', help="Load pre-existing mapping file")
parser.add_argument('-o', '--out_map', help="Save final mapping to file")
args = parser.parse_args()


def cluster(text=None, infile=None, classes=None, class_file=None, class_offset=None, forward_lambda=None, ngram_input=None, min_count=None, out=None, print_freqs=None, quiet=None, refine=None, rev_alternate=None, threads=None, tune_cycles=None, unidirectional=None, verbose=None, word_vectors=None):
    clusters = {}
    cmd_str = "clustercat "

    if (infile):
        cmd_str += " --in " + str(infile)

    if (classes):
        cmd_str += " --classes " + str(classes)

    if (class_file):
        cmd_str += " --class-file " + str(class_file)

    if (class_offset):
        cmd_str += " --class-offset " + str(class_offset)

    if (forward_lambda):
        cmd_str += " --forward-lambda " + str(forward_lambda)

    if (ngram_input):
        cmd_str += " --ngram-input "

    if (min_count):
        cmd_str += " --min-count " + str(min_count)

    if (out):
        cmd_str += " --out " + out

    if (print_freqs):
        cmd_str += " --print-freqs "

    if (quiet):
        cmd_str += " --quiet "

    if (refine):
        cmd_str += " --refine " + str(refine)

    if (rev_alternate):
        cmd_str += " --rev-alternate " + str(rev_alternate)

    if (threads):
        cmd_str += " --threads " + str(threads)

    if (tune_cycles):
        cmd_str += " --tune-cycles " + str(tune_cycles)

    if (unidirectional):
        cmd_str += " --unidirectional "

    if (verbose):
        cmd_str += " --verbose "

    if (word_vectors):
        cmd_str += " --word-vectors " + word_vectors

    #print(cmd_str, file=sys.stderr)  # Use Python 3 interpreter

    cmd_out = subprocess.check_output(cmd_str, shell=True, universal_newlines=True)

    for line in cmd_out.split("\n"):
        split_line = line.split("\t")
        try:
            clusters[split_line[0]] = split_line[1]
        except:
            pass
    return clusters

def main():
    cluster(infile=sys.stdin)

if __name__ == '__main__':
    main()

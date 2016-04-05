#!/usr/bin/env python3
# By Jon Dehdari, 2016
# MIT License
# Simple Python wrapper for ClusterCat

import sys, os, argparse, subprocess, distutils.spawn
parser = argparse.ArgumentParser(description='Converts words to integers, online')

parser.add_argument('-i', '--in', help="Load input training file")
parser.add_argument('-o', '--out', help="Save final mapping to file")
args = parser.parse_args()


def cluster(text=None, in_file=None, classes=None, class_file=None, class_offset=None, forward_lambda=None, ngram_input=None, min_count=None, out=None, print_freqs=None, quiet=None, refine=None, rev_alternate=None, threads=None, tune_cycles=None, unidirectional=None, verbose=None, word_vectors=None):

    # First check to see if we can access clustercat binary relative to this module.  If not, try $PATH.  If not, :-(
    cc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Python 2 doesn't return absolute path in __file__
    cc_bin = os.path.join(cc_dir, 'bin', 'clustercat')
    if (os.path.isfile(cc_bin)):
        cmd_str = [cc_bin]
    elif (distutils.spawn.find_executable("clustercat")):
        cmd_str  = ["clustercat"]
    else:
        print("Error: Unable to access clustercat binary from either ", cc_dir, " or $PATH.  In the parent directory, first run 'make install', and then add $HOME/bin/ to your $PATH, by typing the following command:\necho 'PATH=$PATH:$HOME/bin' >> $HOME/.bashrc  &&  source $HOME/.bashrc")
        exit(1)


    # Now translate function arguments to command-line arguments
    if (in_file):
        cmd_str.append("--in")
        cmd_str.append(str(in_file))

    if (classes):
        cmd_str.append("--classes")
        cmd_str.append(str(classes))

    if (class_file):
        cmd_str.append("--class-file")
        cmd_str.append(str(class_file))

    if (class_offset):
        cmd_str.append("--class-offset")
        cmd_str.append(str(class_offset))

    if (forward_lambda):
        cmd_str.append("--forward-lambda")
        cmd_str.append(str(forward_lambda))

    if (ngram_input):
        cmd_str.append("--ngram-input")

    if (min_count):
        cmd_str.append("--min-count")
        cmd_str.append(str(min_count))

    if (out):
        cmd_str.append("--out")
        cmd_str.append(out)

    if (print_freqs):
        cmd_str.append("--print-freqs")

    if (quiet):
        cmd_str.append("--quiet")

    if (refine):
        cmd_str.append("--refine")
        cmd_str.append(str(refine))

    if (rev_alternate):
        cmd_str.append("--rev-alternate")
        cmd_str.append(str(rev_alternate))

    if (threads):
        cmd_str.append("--threads")
        cmd_str.append(str(threads))

    if (tune_cycles):
        cmd_str.append("--tune-cycles")
        cmd_str.append(str(tune_cycles))

    if (unidirectional):
        cmd_str.append("--unidirectional")

    if (verbose):
        cmd_str.append("--verbose")

    if (word_vectors):
        cmd_str.append("--word-vectors")
        cmd_str.append(word_vectors)

    #print(cmd_str, file=sys.stderr)  # Use Python 3 interpreter

    cmd_out = ''
    if (text and not in_file):
        p1 = subprocess.Popen(["printf", "\n".join(text)], stdout=subprocess.PIPE, universal_newlines=True)
        p2 = subprocess.Popen(cmd_str, stdin=p1.stdout, stdout=subprocess.PIPE, universal_newlines=True)
        p1.stdout.close()
        cmd_out = p2.communicate()[0]
    elif (in_file and not text):
        cmd_out = subprocess.check_output(cmd_str, universal_newlines=True)
    else:
        print("Error: supply either text or in_file argument to clustercat.cluster(), but not both")

    clusters = {}
    for line in cmd_out.split("\n"):
        split_line = line.split("\t")
        try:
            clusters[split_line[0]] = int(split_line[1])
        except:
            pass
    return clusters


def main():
    print(cluster(text=sys.stdin))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# By Jon Dehdari, 2016
# MIT License
""" Fast, flexible word clusters """

import sys, os, argparse
import subprocess, distutils.spawn
parser = argparse.ArgumentParser(description='Clusters words, or tags them')

parser.add_argument('-i', '--in',  help="Load input training file")
parser.add_argument('-o', '--out', help="Save final mapping to file")
parser.add_argument('-t', '--tag', help="Tag stdin input, using clustering in supplied argument")
args = parser.parse_args()

unk = '<unk>'

def load(in_file=None, format='tsv'):
    """ Load a clustering from a file. By default the input file is a tab-separated listing of words and their cluster ID. Returns a dictionary of the clustering. """

    mapping = {}
    if (format == 'tsv'):
        with open(in_file) as f:
            # Primary sort by value (cluster ID), secondary sort by key (word)
            for line in f:
                # Keep the full split line instead of key, val to allow for counts in optional third column
                tokens = line.split()
                mapping[tokens[0]] = int(tokens[1])

    return mapping


def save(mapping=None, out=None, format='tsv'):
    """ Save a clustering (dictionary) to file. By default the output file is a tab-separated listing of words and their cluster ID. """

    if (format == 'tsv'):
        with open(out, 'w') as outfile:
            # Primary sort by value (cluster ID), secondary sort by key (word)
            for key in sorted(sorted(mapping), key=mapping.get):
                line = str(key) + '\t' + str(mapping[key]) + '\n'
                outfile.write(line)


def tag_string(mapping=None, text=None, unk=unk):
    """Tag a string with the corresponding cluster ID's. If a word is not found in the clustering, use unk. Returns a string. """

    newsent = ""
    for word in text.split():
        if word in mapping:
            newsent += ' ' + str(mapping[word])
        elif unk in mapping:
            newsent += ' ' + str(mapping[unk])
        else:
            newsent += ' ' + "<unk>"
    return newsent.lstrip()


def tag_stdin(mapping=None, unk=unk):
    """ This calls tag_string() for each line in stdin, and prints the result to stdout. """

    for line in sys.stdin:
        print(tag_string(mapping=mapping, text=line, unk=unk))


def cluster(text=None, in_file=None, classes=None, class_file=None, class_offset=None, forward_lambda=None, ngram_input=None, min_count=None, out=None, print_freqs=None, quiet=None, refine=None, rev_alternate=None, threads=None, tune_cycles=None, unidirectional=None, verbose=None, word_vectors=None):
    """
    Produce a clustering, given a textual input. There is one required argument (the training input text), and many optional arguments. The one required argument is either text or in_file. The argument text is a list of Python strings. The argument in_file is a path to a text file, consisting of preprocessed (eg. tokenized) one-sentence-per-line text. The use of text is probably not a good idea for large corpora.

    The other optional arguments are described by running the compiled clustercat binary with the --help argument, except that the leading -- from the shell argument is removed, and - is replaced with _. So for example, instead of --tune-cycles 15, the Python function argument would be tune_cycles=15 .

    Returns a dictionary of the form { word : cluster_id } .
    """

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
    clustercat_params = {"in_file": "--in", "out": "--out",
    "classes": "--classes",
    "class_file": "--class-file",
    "class_offset": "--class-offset",
    "forward_lambda": "--forward-lambda",
    "ngram_input": "--ngram-input",
    "min_count": "--min-count",
    "refine": "--refine", 
    "rev_alternate": "--rev-alternate", 
    "threads": "--threads", 
    "tune_cycles": "--tune-cycles",
    "word_vectors": "--word-vectors"}
    
    boolean_params = {
    "print_freqs": "--print-freqs",
    "quiet": "--quiet",
    "unidirectional": "--unidirectional", 
    "verbose": "--verbose"}

    for arg, value in locals().items():
        if arg in boolean_params and value == True: # Check for boolean parameters
            cmd_str.append(boolean_params[arg])
        elif arg in clustercat_params and value is not None: # Other non-boolean parameters that are not None
            cmd_str.append(clustercat_params[arg])
            cmd_str.append(str(value))

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
    if args.tag:
        mapping = load(in_file=args.tag)
        tag_stdin(mapping=mapping)
    else:
        mapping = cluster(text=sys.stdin)
        if args.out:
            save(mapping=mapping, out=args.out)
        else:
            print(mapping)


if __name__ == '__main__':
    main()

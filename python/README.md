# Python ClusterCat


## Installation
First follow the [installation instructions](../README.md) in the above directory.
After that, you normally don't need to install anything here.  You can load the module `clustercat` using either Python 2 or 3.

    cd python
    python3
    >>> import clustercat
    >>> clustering = clustercat.cluster(text=['this is a test', 'that is only a test', 'bye'], min_count=1)
    >>> print(clustering)

If you get an error message saying that it is unable to access clustercat binary, follow all the instructions in the error message.

To import this module from a different directory, you can add the module's directory to `$PYTHONPATH`:

    cd python
	echo "export PYTHONPATH=\$PYTHONPATH:`pwd`" >> ~/.bashrc
	source ~/.bashrc

## Usage
The function `cluster()` is provided in the module `clustercat`.  There is one required argument (the training input text), and many optional arguments.  The one required argument is **either** `text` **or** `in_file`.  The argument `text` is for a list of Python strings.  The argument `in_file` is a path to a text file, consisting of preprocessed (eg. tokenized) one-sentence-per-line text.  The use of `text` is probably not a good idea for large corpora.

```Python
clustercat.cluster(text=['this is a test', 'that is only a test', 'bye'], min_count=1)
clustercat.cluster(in_file='/tmp/corpus.tok.txt', min_count=3)
```

The other optional arguments are described by running the compiled clustercat binary with the `--help` argument, except that the leading `--` from the shell argument is removed, and `-` is replaced with `_`.  So for example, instead of `--tune-cycles 15`, the Python function argument would be `tune_cycles=15`

The function `clustercat.cluster()` returns a dictionary of the form `{ word : cluster_id }` .

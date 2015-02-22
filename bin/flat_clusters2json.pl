#!/usr/bin/env perl
## By Jon Dehdari 2015
## Converts boring tsv clustering format to json for visualization
## Usage: perl clusters2json.pl [options] < in > out

use strict;
use Getopt::Long;

## Defaults
#...

my $usage     = <<"END_OF_USAGE";
clusters2json.pl    (c) 2015 Jon Dehdari - LGPL v3 or Mozilla Public License v2

Usage:    perl $0 [options] < in > out

Function: Converts tsv clustering format to json for visualization

Options:
 -h, --help        Print this usage

END_OF_USAGE

GetOptions(
    'h|help|?'		=> sub { print $usage; exit; },
) or die $usage;

my ($word, $cluster, $freq) = undef;
my $last_cluster = -1;

print <<END;
{
  "name": "Clusters",
  "children": [
    {
END

while (<>) {
	chomp;
	($word, $cluster, $freq) = split;
	$freq or $freq = 1; # if word frequencies aren't provided

	$word =~ s/(["\/])/\\$1/g; # escape problematic characters
	#$word =~ s/</&lt;/g;
	#$word =~ s/>/&gt;/g;

	if ($cluster != $last_cluster) { # We've reached a new cluster

		if ($last_cluster != -1) { # end cluster's children (ie words), then start new cluster
			print <<END
}
      ]
    },
    {
END
		}
		$last_cluster = $cluster;
		print <<END;
      "name": "$cluster",
      "children": [
END
		print "        {";
	} else { # within a cluster
		print "},\n        {";
	}

	print "\"name\": \"$word\", \"size\": $freq";
} # end while (<>) loop

print <<END;
}
      ]
    }
  ]
}
END

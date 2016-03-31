%module clustercat

%{
#include "clustercat.h"						// Model importing/exporting functions
#include "clustercat-array.h"				// which_maxf()
#include "clustercat-data.h"
#include "clustercat-cluster.h"				// cluster()
#include "clustercat-dbg.h"					// for printing out various complex data structures
#include "clustercat-import-class-file.h"	// import_class_file()
#include "clustercat-io.h"					// process_input()
#include "clustercat-math.h"				// perplexity(), powi()
%}

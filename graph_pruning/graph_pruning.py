import csv
import sys
import methods.m_old as old
import methods.m_tarjan as tarjan
import methods.m_dfs as dfs
import methods.m_mfas as mfas
import methods.m_hierarchy as hierarchy

########################################################################################################################
#
# Methods
#
########################################################################################################################
methods = {
    "old": old,
    "tarjan": tarjan,
    "dfs": dfs,
    "mfas": mfas,

    "hierarchy_ensemble_greedy":  hierarchy,
    "hierarchy_ensemble_forward": hierarchy,
    "hierarchy_ensemble_backward": hierarchy,
    "hierarchy_ensemble_voting":  hierarchy,

    "hierarchy_pagerank_greedy": hierarchy,
    "hierarchy_pagerank_forward": hierarchy,
    "hierarchy_pagerank_backward": hierarchy,
    "hierarchy_pagerank_voting": hierarchy,

    "hierarchy_socialagony_greedy": hierarchy,
    "hierarchy_socialagony_forward": hierarchy,
    "hierarchy_socialagony_backward": hierarchy,
    "hierarchy_socialagony_voting": hierarchy,

    "hierarchy_trueskill_greedy": hierarchy,
    "hierarchy_trueskill_forward": hierarchy,
    "hierarchy_trueskill_backward": hierarchy,
    "hierarchy_trueskill_voting": hierarchy
}

########################################################################################################################
#
# Main
#
########################################################################################################################
filename_in = None
filename_out = None
gephi_out = None
delimiter = '\t'
mode = "old"

if len(sys.argv) == 1:
    print("graph_pruning.py input_csv output_csv [mode] [gephi_output_file] [csv_delimeter]")
    print("   mode              : Supported: %s. Default=%s" % (sorted(methods.keys()), mode))
    print("   gephi_output_file : Output file for easier import to gephi; not supported by all modes. Default no file.")
    print("   csv_delimeter     : Delimiter of CSV-files. Default=%s" % delimiter)
    exit(1)

if len(sys.argv) >= 2:
    filename_in = sys.argv[1]

if len(sys.argv) >= 3:
    filename_out = sys.argv[2]

if len(sys.argv) >= 4:
    mode = sys.argv[3]

if len(sys.argv) >= 5:
    gephi_out = sys.argv[4]

if len(sys.argv) >= 6:
    delimiter = sys.argv[5]


if filename_in is None:
    raise Exception("No CSV-file provided")

if filename_out is None:
    raise Exception("No output provided")

print("Reading file: %s" % filename_in)
print("Output file: %s" % filename_out)

if gephi_out is not None and gephi_out != "":
    print("Gephi file: %s" % gephi_out)

print("Delimiter: %s" % delimiter)
print("Mode: %s" % mode)

if mode not in methods:
    raise Exception("Mode '%s' unknown. Supported: %s" % (mode, sorted(methods.keys())))

method = methods[mode]

with open(filename_in, "r") as f:
    reader = csv.reader(f, delimiter=delimiter)

    for i, line in enumerate(reader):
        method.prepare(line)

cycles_removed = method.do(filename_out, delimiter, mode, gephi_out)
print("Graph pruning finished.")
print("Removed: %s" % cycles_removed)

import random
import methods.util.write_graph
from tarjan import tarjan

hyper_to_hypo = {}


def prepare(node):
    id = node[0]
    hypo = node[1]
    hyper = node[2]

    # print("Read line with ID '%s': %s --> %s" % (id, hypo, hyper))

    if hyper not in hyper_to_hypo:
        hyper_to_hypo[hyper] = []

    if hypo not in hyper_to_hypo:
        hyper_to_hypo[hypo] = []

    hyper_to_hypo[hyper].append(hypo)


def do(filename_out, delimiter, mode, gephi_out):
    cycles_removed = 0
    cycle = []  # Initialize with value to trigger the while loop. Python has no do-while...

    while (cycle is not None):
        t = tarjan(hyper_to_hypo)

        # ['plant pathology','pathology'],

        i = 0
        cycle = None

        while (cycle is None and i < len(t)):
            if len(t[i]) > 1:
                # Do pruning
                print("Cycle detected: %s" % t[i])
                cycle = t[i]

            i += 1

        if cycle is not None:
            hypernym_index_removed_from = random.randint(0, len(cycle) - 1)
            hypernym_removed_from = cycle[hypernym_index_removed_from]

            for c in cycle:
                if c in hyper_to_hypo[hypernym_removed_from]:
                    print("Remove hyponym '%s' from hypernym '%s'." % (c, hypernym_removed_from))
                    hyper_to_hypo[hypernym_removed_from].remove(c)
                    cycles_removed += 1
                    break

    print("Removed %s cycles." % cycles_removed)
    methods.util.write_graph.hyper_to_hypo_graph(filename_out, hyper_to_hypo, gephi_out=gephi_out, delimiter=delimiter)
    return cycles_removed

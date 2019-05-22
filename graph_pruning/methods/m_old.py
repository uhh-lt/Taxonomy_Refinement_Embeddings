
import random
import csv

hypo_to_hyper = {}


def prepare(node):
    id = node[0]
    hypo = node[1]
    hyper = node[2]

    #print("Read line with ID '%s': %s --> %s" % (id, hypo, hyper))

    if hypo not in hypo_to_hyper:
        hypo_to_hyper[hypo] = hyper
    elif random.randint(0, 1) == 0:
        print("Replace hypernym of hyponym '%s' with: %s" % (hypo, hyper))

        hypo_to_hyper[hypo] = hyper
    else:
        print("Keep hypernym '%s' of hyponym '%s'." % (hypo_to_hyper[hypo], hypo))



def do(filename_out, delimiter, mode):
    print("Relations created (hypo_to_hyper):")
    print(hypo_to_hyper)

    print("Write result.")

    with open(filename_out, "w+") as f:
        writer = csv.writer(f, delimiter=delimiter)
        id = 0

        for hypo in hypo_to_hyper:
            row = [id, hypo, hypo_to_hyper[hypo]]
            writer.writerow(row)

            id += 1

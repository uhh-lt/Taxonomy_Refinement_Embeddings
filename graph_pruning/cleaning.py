import os
import csv
import random
import sys

def remove_cycles(list_data):
    hypo_to_hyper = {}
    for element in list_data:
        id = element[0]
        hypo = element[1]
        hyper = element[2]

        print("Read line with ID '%s': %s --> %s" % (id, hypo, hyper))

        if hypo not in hypo_to_hyper:
            hypo_to_hyper[hypo] = hyper
        elif random.randint(0, 1) == 0:
            print("Replace hypernym of hyponym '%s' with: %s" % (hypo, hyper))
            hypo_to_hyper[hypo] = hyper
        else:
            print("Keep hypernym '%s' of hyponym '%s'." % (hypo_to_hyper[hypo], hypo))

    return hypo_to_hyper


#Returns if contected to root or the highest parent
def connected_to_root(element, list_data, root):
    parent = element
    parent_last = None
    while parent != parent_last:
        parent_last = parent
        for relation in list_data:
            if parent == relation[1]:
                parent = relation[2]
    return [parent == root, parent]



def zero_out_nodes(elements):
    zero_out = set([])
    for element in elements:
        parents = [relation[2] for relation in list_data]
        if not element in parents:
            zero_out.add(element)
    return zero_out


#Used for removing everythign above root
def all_parent_rel(element, list_data):
    parent_root = element
    remove_relations = []
    for relation in list_data:
        if relation[1] == parent_root:
            remove_relations.append(relation)
    for ele in remove_relations:
        if ele[1] != ele[2]:
            remove_relations += all_parent_rel(ele[2], list_data)
    return set(remove_relations)



filename_in = None
filename_out = None
ROOT = None

if len(sys.argv) >= 2:
    filename_in = sys.argv[1]

if len(sys.argv) >= 3:
    filename_out = sys.argv[2]

if len(sys.argv) >= 4:
    ROOT = sys.argv[3]


if filename_in is None:
    raise Exception("No CSV-file provided")

if filename_out is None:
    raise Exception("No output provided")

list_data = []
iter = 1
with open(filename_in, 'r') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data.append((line[0], line[1], line[2]))
        iter +=1


all_parent_root = all_parent_rel(ROOT, list_data)
for el in all_parent_root:
    list_data.remove(el)

elements = set([])
for relation in list_data:
    elements.add(relation[1])
    elements.add(relation[2])
elements_connected = set([])
#make sure that every element is transitively connected to the root, otherwise add direct link to root
for element in elements:
    ele_root = connected_to_root(element, list_data, ROOT)
    if not ele_root[0] and element == ele_root[1]:
       print(element,  end=' ')
       list_data.append((iter, ele_root[1], ROOT))
       iter+=1
print('\n')

with open(filename_out, 'w') as f:
    for element in list_data:
        f.write(str(element[0]) + '\t' + element[1] + '\t' + element[2]  + '\n')

import networkx as nx
import numpy as np
import sys
import methods.util.write_graph as write_graph
import methods.util.util as util

sys.setrecursionlimit(5500000)

g = nx.DiGraph()


def prepare(line):
    g.add_edge(line[1], line[2])


def do(filename_out, delimiter, mode, gephi_out):
    edges_to_be_removed = dfs_remove_back_edges()

    cycles_removed = util.remove_edges_from_network_graph(g, edges_to_be_removed)
    write_graph.network_graph(filename_out, g, gephi_out=gephi_out, delimiter=delimiter)

    return cycles_removed

def dfs_remove_back_edges():
    '''
    0: white, not visited
    1: grey, being visited
    2: black, already visited
    '''
    # g = nx.read_edgelist(graph_file,create_using = ,delimiter='\t')
    nodes_color = {}
    edges_to_be_removed = []
    for node in g:
        nodes_color[node] = 0

    nodes_order = list(g)
    nodes_order = np.random.permutation(nodes_order)
    num_dfs = 0
    for node in nodes_order:

        if nodes_color[node] == 0:
            num_dfs += 1
            dfs_visit_recursively(g, node, nodes_color, edges_to_be_removed)

    return edges_to_be_removed


def dfs_visit_recursively(g, node, nodes_color, edges_to_be_removed):
    nodes_color[node] = 1
    nodes_order = list(g.successors(node))
    nodes_order = np.random.permutation(nodes_order)
    for child in nodes_order:
        if nodes_color[child] == 0:
            dfs_visit_recursively(g, child, nodes_color, edges_to_be_removed)
        elif nodes_color[child] == 1:
            edges_to_be_removed.append((node, child))

    nodes_color[node] = 2

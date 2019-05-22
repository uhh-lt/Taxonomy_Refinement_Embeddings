import networkx as nx
import numpy as np
import methods.util.write_graph as write_graph
import methods.util.util as util

g = nx.DiGraph()


def prepare(line):
    g.add_edge(line[1], line[2])


def do(filename_out, delimiter, mode, gephi_out):
    edges_to_be_removed = remove_cycle_edges_by_mfas()

    cycles_removed = util.remove_edges_from_network_graph(g, edges_to_be_removed)
    write_graph.network_graph(filename_out, g, gephi_out=gephi_out, delimiter=delimiter)
    return cycles_removed



def pick_from_dict(d, order="max"):
    min_k, min_v = 0, 10000

    min_items = []
    max_k, max_v = 0, -10000

    max_items = []
    for k, v in d.iteritems():
        if v > max_v:
            max_v = v
            max_items = [(k, max_v)]
        elif v == max_v:
            max_items.append((k, v))

        if v < min_v:
            min_v = v
            min_items = [(k, min_v)]
        elif v == min_v:
            min_items.append((k, v))

    max_k, max_v = pick_randomly(max_items)
    min_k, min_v = pick_randomly(min_items)

    if order == "max":
        return max_k, max_v
    if order == "min":
        return min_k, min_v
    else:
        return max_k, max_v, min_k, min_v


def pick_randomly(source):
    np.random.shuffle(source)
    np.random.shuffle(source)
    np.random.shuffle(source)
    return source[0]


def filter_big_scc(g, edges_to_be_removed):
    # Given a graph g and edges to be removed
    # Return a list of big scc subgraphs (# of nodes >= 2)
    g.remove_edges_from(edges_to_be_removed)
    sub_graphs = filter(lambda scc: scc.number_of_nodes() >= 2, nx.strongly_connected_component_subgraphs(g))
    return sub_graphs


def get_big_sccs(g):
    self_loop_edges = g.selfloop_edges()
    g.remove_edges_from(g.selfloop_edges())
    num_big_sccs = 0
    edges_to_be_removed = []
    big_sccs = []
    for sub in nx.strongly_connected_component_subgraphs(g):
        number_of_nodes = sub.number_of_nodes()
        if number_of_nodes >= 2:
            # strongly connected components
            num_big_sccs += 1
            big_sccs.append(sub)
    # print(" # big sccs: %d" % (num_big_sccs))
    return big_sccs


def nodes_in_scc(sccs):
    scc_nodes = []
    scc_edges = []
    for scc in sccs:
        scc_nodes += list(scc.nodes())
        scc_edges += list(scc.edges())

    # print("# nodes in big sccs: %d" % len(scc_nodes))
    # print("# edges in big sccs: %d" % len(scc_edges))
    return scc_nodes


def scc_nodes_edges(g):
    scc_nodes = set()
    scc_edges = set()
    num_big_sccs = 0
    num_nodes_biggest_scc = 0
    biggest_scc = None
    for sub in nx.strongly_connected_component_subgraphs(g):
        number_nodes = sub.number_of_nodes()
        if number_nodes >= 2:
            scc_nodes.update(sub.nodes())
            scc_edges.update(sub.edges())
            num_big_sccs += 1
            if num_nodes_biggest_scc < number_nodes:
                num_nodes_biggest_scc = number_nodes
                biggest_scc = sub
    nonscc_nodes = set(g.nodes()) - scc_nodes
    nonscc_edges = set(g.edges()) - scc_edges
    print("num nodes biggest scc: %d" % num_nodes_biggest_scc)
    print("num of big sccs: %d" % num_big_sccs)
    if biggest_scc == None:
        return scc_nodes, scc_nodes, nonscc_nodes, nonscc_edges
    print("# nodes in biggest scc: %d, # edges in biggest scc: %d" % (
    biggest_scc.number_of_nodes(), biggest_scc.number_of_edges()))
    print("# nodes,edges in scc: (%d,%d), # nodes, edges in non-scc: (%d,%d) " % (
    len(scc_nodes), len(scc_edges), len(nonscc_nodes), len(nonscc_edges)))
    num_of_nodes = g.number_of_nodes()
    num_of_edges = g.number_of_edges()
    print(
                "# nodes in graph: %d, # of edges in graph: %d, percentage nodes, edges in scc: (%0.4f,%0.4f), percentage nodes, edges in non-scc: (%0.4f,%0.4f)" % (
        num_of_nodes, num_of_edges, len(scc_nodes) * 1.0 / num_of_nodes, len(scc_edges) * 1.0 / num_of_edges,
        len(nonscc_nodes) * 1.0 / num_of_nodes, len(nonscc_edges) * 1.0 / num_of_edges))
    return scc_nodes, scc_edges, nonscc_nodes, nonscc_edges


def get_nodes_degree_dict(g, nodes):
    # get nodes degree dict: key = node, value = (max(d(in)/d(out),d(out)/d(in),"in" or "out")
    in_degrees = g.in_degree(nodes)
    out_degrees = g.out_degree(nodes)
    degree_dict = {}
    for node in nodes:
        in_d = in_degrees[node]
        out_d = out_degrees[node]
        if in_d >= out_d:
            try:
                value = in_d * 1.0 / out_d
            except Exception as e:
                value = 0
            f = "in"
        else:
            try:
                value = out_d * 1.0 / in_d
            except Exception as e:
                value = 0
            f = "out"
        degree_dict[node] = (value, f)
    # print("node: %d: %s" % (node,degree_dict[node]))
    return degree_dict


def greedy_local_heuristic(sccs, degree_dict, edges_to_be_removed):
    while True:
        graph = sccs.pop()
        temp_nodes_degree_dict = {}
        for node in graph.nodes():
            temp_nodes_degree_dict[node] = degree_dict[node][0]

        max_node, _ = pick_from_dict(temp_nodes_degree_dict)
        max_value = degree_dict[max_node]
        # degrees = [(node,degree_dict[node]) for node in list(graph.nodes())]
        # max_node,max_value = max(degrees,key = lambda x: x[1][0])
        if max_value[1] == "in":
            # indegree > outdegree, remove out-edges
            edges = [(max_node, o) for o in graph.neighbors(max_node)]
        else:
            # outdegree > indegree, remove in-edges
            edges = [(i, max_node) for i in graph.predecessors(max_node)]
        edges_to_be_removed += edges
        sub_graphs = filter_big_scc(graph, edges_to_be_removed)
        if sub_graphs:
            for index, sub in enumerate(sub_graphs):
                sccs.append(sub)
        if not sccs:
            return


def remove_self_loops_from_graph(g):
    self_loops = list(g.selfloop_edges())
    g.remove_edges_from(self_loops)
    return self_loops


def remove_cycle_edges_by_mfas():
    self_loops = remove_self_loops_from_graph(g)

    scc_nodes, _, _, _ = scc_nodes_edges(g)
    degree_dict = get_nodes_degree_dict(g, scc_nodes)
    sccs = get_big_sccs(g)
    if len(sccs) == 0:
        print("After removal of self loop edgs: %s" % nx.is_directed_acyclic_graph(g))
        return self_loops
    edges_to_be_removed = []
    import timeit
    t1 = timeit.default_timer()
    greedy_local_heuristic(sccs, degree_dict, edges_to_be_removed)
    t2 = timeit.default_timer()
    print("mfas time usage: %0.4f s" % (t2 - t1))
    edges_to_be_removed = list(set(edges_to_be_removed))

    # g.remove_edges_from(edges_to_be_removed)
    edges_to_be_removed += self_loops

    return edges_to_be_removed

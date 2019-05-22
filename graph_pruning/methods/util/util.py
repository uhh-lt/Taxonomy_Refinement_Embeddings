



def remove_edges_from_network_graph(g, edges_to_remove):
    cycles_removed = 0

    for edge in edges_to_remove:
        print("Remove edge: (%s, %s)" % edge)
        g.remove_edge(edge[0], edge[1])
        cycles_removed += 1

    print("Edges removed: %s" % len(edges_to_remove))
    return cycles_removed

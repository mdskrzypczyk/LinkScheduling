import networkx as nx
from esss import swap_links, distill_links, esss, Protocol


def create_protocol(path, G, Fmin, Rmin):
    def filter_node(node):
        return node in path

    def filter_edge(node1, node2):
        return node1 in path and node2 in path

    print("Creating protocol on path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
    subG = nx.subgraph_view(G, filter_node, filter_edge)
    protocol = esss(path, subG, Fmin, Rmin)
    if type(protocol) != Protocol and protocol is not None:
        return protocol
    else:
        return None


def swap_then_distill(F1, F2, n):
    Fs = swap_links(F1, F2)
    for i in range(1, n):
        Fs = distill_links(Fs, Fs)

    return Fs


def distill_then_swap(F1, F2, n):
    Fl = F1
    Fr = F2
    for i in range(1, n):
        Fl = distill_links(Fl, F1)
        Fr = distill_links(Fr, F2)

    return swap_links(Fl, Fr)


def compare():
    F1 = 0.65
    F2 = 0.8
    for i in range(1, 20):
        print("STD: {}".format(swap_then_distill(F1, F2, i)))
        print("DTS: {}".format(distill_then_swap(F1, F2, i)))

import networkx as nx
from device_characteristics.nv_links import load_link_data
from jobscheduling.haversine import distance
from math import sqrt

SURFNET_GML = "Surfnet.gml"

def get_dimensions(n):
    divisors = []
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
            divisors.append(currentDiv+1)

    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i]-sqrt(n)))
    wIndex = hIndex + 1

    return divisors[hIndex], divisors[wIndex]


def gen_topologies(n, num_comm_q=2, num_storage_q=2, link_distance=5):
    d_to_cap = load_link_data()
    link_capabilities = [(d, d_to_cap[str(d)]) for d in [5]]
    link_capability = d_to_cap[str(link_distance)]
    # Line
    lineGcq = nx.Graph()
    lineG = nx.Graph()
    for i in range(n):
        node = "{}".format(i)
        comm_qs = []
        storage_qs = []
        for c in range(num_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        lineGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        lineG.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        lineG.nodes[node]["end_node"] = True
        if i > 0:
            prev_node_id = i - 1
            for j in range(num_comm_q):
                for k in range(num_comm_q):
                    lineGcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(i, k))

            lineG.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability, weight=link_distance)

    # Ring
    ringGcq = nx.Graph()
    ringG = nx.Graph()
    for i in range(n):
        node = "{}".format(i)
        comm_qs = []
        storage_qs = []
        for c in range(num_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        ringGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        ringG.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        ringG.nodes[node]["end_node"] = True
        if i > 0:
            prev_node_id = i - 1
            for j in range(num_comm_q):
                for k in range(num_comm_q):
                    ringGcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(i, k))
            ringG.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability, weight=link_distance)

    ringG.add_edge("{}".format(0), "{}".format(n-1), capabilities=link_capability, weight=link_distance)
    for j in range(1):
        for k in range(1):
            ringGcq.add_edge("{}-{}".format(0, j), "{}-{}".format(n-1, k), capabilities=link_capability, weight=link_distance)

    # Demo
    demoGcq = nx.Graph()
    demoG = nx.Graph()
    for i in range(4):
        comm_qs = []
        storage_qs = []
        for c in range(num_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        demoGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        demoG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)
        if i > 0:
            prev_node_id = i - 1
            for j in range(num_comm_q):
                for k in range(num_comm_q):
                    demoGcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(i, k))

    demoG.add_edge("0", "1", capabilities=d_to_cap["10"], weight=10)
    demoG.add_edge("1", "2", capabilities=d_to_cap["15"], weight=15)
    demoG.add_edge("2", "3", capabilities=d_to_cap["35"], weight=35)
    demoG.add_edge("3", "0", capabilities=d_to_cap["50"], weight=50)

    # Grid
    w, h = get_dimensions(n)
    gridGcq = nx.Graph()
    gridG = nx.Graph()
    for i in range(w):
        for j in range(h):
            comm_qs = []
            storage_qs = []
            for c in range(num_comm_q):
                comm_q_id = "{},{}-C{}".format(i, j, c)
                comm_qs.append(comm_q_id)
            for s in range(num_storage_q):
                storage_q_id = "{},{}-S{}".format(i, j, s)
                storage_qs.append(storage_q_id)
            gridGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
            gridG.add_node("{},{}".format(i, j), comm_qs=comm_qs, storage_qs=storage_qs)

            # Connect upward
            if j > 0:
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i, j-1), capabilities=link_capability,
                               weight=link_distance)
                for k in range(num_comm_q):
                    for l in range(num_comm_q):
                        gridGcq.add_edge("{},{}-C{}".format(i, j-1, k), "{},{}-C{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance)
            # Connect left
            if i > 0:
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i - 1, j), capabilities=link_capability,
                               weight=link_distance)
                for k in range(num_comm_q):
                    for l in range(num_comm_q):
                        gridGcq.add_edge("{},{}-C{}".format(i-1, j, k), "{},{}-C{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance)

    return [(lineGcq, lineG), (ringGcq, ringG), (gridGcq, gridG), (demoGcq, demoG)]


def gen_plus_topology(num_nodes=5, end_node_resources=(1, 3), center_resources=(1, 3), link_distance=5):
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_distance)]
    # Line
    starGcq = nx.Graph()
    starG = nx.Graph()

    # First make the center
    num_comm_center, num_storage_center = center_resources
    comm_qs = []
    storage_qs = []
    i = num_nodes - 1
    for c in range(num_comm_center):
        comm_q_id = "{}-C{}".format(i, c)
        comm_qs.append(comm_q_id)
    for s in range(num_storage_center):
        storage_q_id = "{}-S{}".format(i, s)
        storage_qs.append(storage_q_id)
    starGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
    starG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)

    # Then make the end nodes
    num_comm_end_node, num_storage_end_node = end_node_resources
    for i in range(num_nodes - 1):
        comm_qs = []
        storage_qs = []
        for c in range(num_comm_end_node):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_end_node):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        starGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        starG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)

        center_node_id = num_nodes - 1
        for j in range(num_comm_center):
            for k in range(num_comm_end_node):
                starGcq.add_edge("{}-C{}".format(center_node_id, j), "{}-C{}".format(i, k))

        starG.add_edge("{}".format(center_node_id), "{}".format(i), capabilities=link_capability,
                       weight=link_distance)

    return starGcq, starG


def gen_grid_topology(num_nodes=9, end_node_resources=(1, 3), repeater_resources=(1, 3), link_distance=5):
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_distance)]
    # Line
    gridGcq = nx.Graph()
    gridG = nx.Graph()

    end_nodes = ['0', '2', '6', '8']
    repeater_nodes = ['1', '3', '4', '5', '7']

    # Then make the end nodes
    for i in range(num_nodes):
        comm_qs = []
        storage_qs = []
        num_comm_qs, num_storage_qs = end_node_resources if str(i) in end_nodes else repeater_resources
        for c in range(num_comm_qs):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_qs):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        gridGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        gridG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)

    edges = [('0', '1'), ('0', '3'),
             ('1', '4'), ('1', '2'),
             ('2', '5'),
             ('3', '6'), ('3', '4'),
             ('4', '7'), ('4', '5'),
             ('5', '8'),
             ('6', '7'), ('7', '8')]

    for edge in edges:
        node1, node2 = edge
        gridG.add_edge("{}".format(node1), "{}".format(node2), capabilities=link_capability, weight=link_distance)
        node1_comms = end_node_resources[0] if node1 in end_nodes else repeater_resources[0]
        node2_comms = end_node_resources[0] if node2 in end_nodes else repeater_resources[0]
        for j in range(node1_comms):
            for k in range(node2_comms):
                gridGcq.add_edge("{}-C{}".format(node1, j), "{}-C{}".format(node2, k))

    return gridGcq, gridG

def load_surfnet():
    # Load graph data
    G = nx.read_gml(SURFNET_GML)

    # Compute edge distances
    for edge in G.edges:
        node1, node2 = edge
        lat1 = G.nodes[node1]["Latitude"]
        lon1 = G.nodes[node1]["Longitude"]
        lat2 = G.nodes[node2]["Latitude"]
        lon2 = G.nodes[node2]["Longitude"]
        d = distance((lat1, lon1), (lat2, lon2))
        G.edges[edge]["length"] = d

    edges_to_remove = [e for e in G.edges if G.edges[e]["length"] > 50]
    G.remove_edges_from(edges_to_remove)
    nodes_to_remove = [n for n in G.nodes if G.degree[n] == 0]
    G.remove_nodes_from(nodes_to_remove)

    return G

def gen_surfnet_topology(end_node_resources, repeater_node_resources):
    G = load_surfnet()
    Gcq = nx.Graph()
    end_nodes = [n for n in G.nodes if G.degree[n] > 2 or G.degree == 1]
    repeater_nodes = [n for n in G.nodes if n not in end_nodes]

    end_node_comms, end_node_storage = end_node_resources
    for node in end_nodes:
        G.nodes[node]["end_node"] = True
        comm_qs = []
        storage_qs = []
        for c in range(end_node_comms):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(end_node_storage):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)

    repeater_node_comms, repeater_node_storage = repeater_node_resources
    for node in repeater_nodes:
        G.nodes[node]["end_node"] = False
        comm_qs = []
        storage_qs = []
        for c in range(repeater_node_comms):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(repeater_node_storage):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)

    for edge in G.edges:
        node1, node2 = edge
        node1_comms = end_node_resources[0] if node1 in end_nodes else repeater_node_resources[0]
        node2_comms = end_node_resources[0] if node1 in end_nodes else repeater_node_resources[0]
        for i in range(node1_comms):
            for j in range(node2_comms):
                Gcq.add_edge("{}-C{}".format(node1, i), "{}-C{}".format(node2, j))

    return Gcq, G



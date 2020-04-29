import networkx as nx
from device_characteristics.nv_links import load_link_data
from math import sqrt


def get_dimensions(n):
    divisors = []
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
            divisors.append(currentDiv + 1)

    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i] - sqrt(n)))
    wIndex = hIndex + 1

    return divisors[hIndex], divisors[wIndex]


def gen_topologies(n, num_comm_q=2, num_storage_q=2, link_distance=5):
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_distance)]

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
            ringG.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability,
                           weight=link_distance)

    ringG.add_edge("{}".format(0), "{}".format(n - 1), capabilities=link_capability, weight=link_distance)
    for j in range(1):
        for k in range(1):
            ringGcq.add_edge("{}-{}".format(0, j), "{}-{}".format(n - 1, k), capabilities=link_capability,
                             weight=link_distance)

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
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i, j - 1), capabilities=link_capability,
                               weight=link_distance)
                for k in range(num_comm_q):
                    for l in range(num_comm_q):
                        gridGcq.add_edge("{},{}-C{}".format(i, j - 1, k), "{},{}-C{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance)
            # Connect left
            if i > 0:
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i - 1, j), capabilities=link_capability,
                               weight=link_distance)
                for k in range(num_comm_q):
                    for l in range(num_comm_q):
                        gridGcq.add_edge("{},{}-C{}".format(i - 1, j, k), "{},{}-C{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance)

    return [(ringGcq, ringG), (gridGcq, gridG), (demoGcq, demoG)]


def gen_line_topology(num_nodes=5, num_comm_q=1, num_storage_q=3, link_distance=5):
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_distance)]
    # Line
    lineGcq = nx.Graph()
    lineG = nx.Graph()
    for i in range(num_nodes):
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

            lineG.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability,
                           weight=link_distance)

    return lineGcq, lineG


def gen_grid_topology(num_nodes=9, end_node_resources=(1, 3), repeater_resources=(1, 3), link_distance=5):
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_distance)]
    # Line
    gridGcq = nx.Graph()
    gridG = nx.Graph()

    end_nodes = ['0', '2', '6', '8']

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

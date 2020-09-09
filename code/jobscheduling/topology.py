import networkx as nx
from device_characteristics.nv_links import load_link_data


def gen_H_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3,
                   link_length=5):
    """
    Generates an H shaped topology of a quantum network.
    :param num_end_node_comm_q: type int
        Number of communication qubits each end node has
    :param num_end_node_storage_q: type int
        Number of storage qubits each end node has
    :param num_rep_comm_q: type int
        Number of communication qubits each repeater has
    :param num_rep_storage_q: type int
        Number of storage qubits each repeater has
    :param link_length: type int
        Length of each link connecting any two nodes
    :return: type tuple
        Tuple of networkx.Graphs that represent the communication resources in the network and the connectivity of nodes
        in the network along with the link capabilities of each link.
    """
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_length)]
    # Line
    Gcq = nx.Graph()
    G = nx.Graph()

    end_nodes = ['0', '2', '3', '5']
    repeater_nodes = ['1', '4']
    edges = [
        ('0', '1'),
        ('1', '2'),
        ('1', '4'),
        ('3', '4'),
        ('4', '5')
    ]

    for node in end_nodes:
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        G.nodes[node]["end_node"] = True

    for node in repeater_nodes:
        comm_qs = []
        storage_qs = []
        for c in range(num_rep_comm_q):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(num_rep_storage_q):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        G.nodes[node]["end_node"] = False

    for node1, node2 in edges:
        G.add_edge("{}".format(node1), "{}".format(node2), capabilities=link_capability, weight=link_length)

    for node1, node2 in edges:
        num_comm_node1 = num_end_node_comm_q if G.nodes[node1]["end_node"] else num_rep_comm_q
        num_comm_node2 = num_end_node_comm_q if G.nodes[node2]["end_node"] else num_rep_comm_q
        for j in range(num_comm_node1):
            for k in range(num_comm_node2):
                Gcq.add_edge("{}-C{}".format(node1, j), "{}-C{}".format(node2, k))

    return Gcq, G


def gen_star_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3,
                      link_length=5):
    """
        Generates an star shaped topology of a quantum network.
        :param num_end_node_comm_q: type int
            Number of communication qubits each end node has
        :param num_end_node_storage_q: type int
            Number of storage qubits each end node has
        :param num_rep_comm_q: type int
            Number of communication qubits each repeater has
        :param num_rep_storage_q: type int
            Number of storage qubits each repeater has
        :param link_length: type int
            Length of each link connecting any two nodes
        :return: type tuple
            Tuple of networkx.Graphs that represent the communication resources in the network and the connectivity of
            nodes in the network along with the link capabilities of each link.
        """
    num_nodes = 4
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_length)]

    # Line
    Gcq = nx.Graph()
    G = nx.Graph()

    # First make the center
    comm_qs = []
    storage_qs = []
    i = num_nodes - 1
    for c in range(num_rep_comm_q):
        comm_q_id = "{}-C{}".format(i, c)
        comm_qs.append(comm_q_id)
    for s in range(num_rep_storage_q):
        storage_q_id = "{}-S{}".format(i, s)
        storage_qs.append(storage_q_id)
    Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
    G.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs, end_node=False)

    # Then make the end nodes
    for i in range(num_nodes - 1):
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        G.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs, end_node=True)

        center_node_id = num_nodes - 1
        for j in range(num_rep_comm_q):
            for k in range(num_end_node_comm_q):
                Gcq.add_edge("{}-C{}".format(center_node_id, j), "{}-C{}".format(i, k))

        G.add_edge("{}".format(center_node_id), "{}".format(i), capabilities=link_capability, weight=link_length)

    return Gcq, G


def gen_line_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3,
                      link_length=5):
    """
        Generates an line shaped topology of a quantum network.
        :param num_end_node_comm_q: type int
            Number of communication qubits each end node has
        :param num_end_node_storage_q: type int
            Number of storage qubits each end node has
        :param num_rep_comm_q: type int
            Number of communication qubits each repeater has
        :param num_rep_storage_q: type int
            Number of storage qubits each repeater has
        :param link_length: type int
            Length of each link connecting any two nodes
        :return: type tuple
            Tuple of networkx.Graphs that represent the communication resources in the network and the connectivity of
            nodes in the network along with the link capabilities of each link.
        """
    num_nodes = 6
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_length)]
    # Line
    Gcq = nx.Graph()
    G = nx.Graph()

    for i in range(num_nodes):
        node = "{}".format(i)
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs, end_node=True)
        if i > 0:
            prev_node_id = i - 1
            for j in range(num_end_node_comm_q):
                for k in range(num_end_node_storage_q):
                    Gcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(i, k))

            G.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability, weight=link_length)

    return Gcq, G


def gen_symm_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3,
                      link_length=5):
    num_nodes = 8
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_length)]

    Gcq = nx.Graph()
    G = nx.Graph()

    # Nodes 0-3 are the end nodes
    for i in range(4):
        node = "{}".format(i)
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs, end_node=True)

    # Nodes 4-7 form the complete internal repeater network
    for i in range(4, 8):
        node = "{}".format(i)
        comm_qs = []
        storage_qs = []
        for c in range(num_rep_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_rep_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs, end_node=False)

    # Internal complete network
    repeater_connections = {
        "4": ["0", "5", "6", "7"],
        "5": ["1", "4", "6", "7"],
        "6": ["2", "4", "5", "7"],
        "7": ["3", "4", "5", "6"]
    }
    for node, connected_nodes in repeater_connections.items():
        for prev_node_id in connected_nodes:
            for j in range(num_end_node_comm_q):
                for k in range(num_end_node_storage_q):
                    Gcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(node, k))

            G.add_edge("{}".format(prev_node_id), "{}".format(node), capabilities=link_capability, weight=link_length)

    return Gcq, G

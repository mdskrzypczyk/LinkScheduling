import networkx as nx
from device_characteristics.nv_links import load_link_data
from fiber_data.graph import construct_graph


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
    # Get link distance to capability data
    d_to_cap = load_link_data()

    # Networks have uniform link lengths so just graph the one set of capabilities
    link_capability = d_to_cap[str(link_length)]

    # Make the resource graph and node graph
    Gcq = nx.Graph()
    G = nx.Graph()

    # Label end nodes and repeaters
    end_nodes = ['0', '2', '3', '5']
    repeater_nodes = ['1', '4']

    # Link pairs of nodes to form H topology
    edges = [
        ('0', '1'),
        ('1', '2'),
        ('1', '4'),
        ('3', '4'),
        ('4', '5')
    ]

    # For each end node create the set of communication and storage qubits
    # Each communication qubit identified by "<NODEID>-C<ENUM>"
    # Each storage qubit identified by "<NODEID>-S<ENUM>"
    for node in end_nodes:
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)

        # Add communication qubits and associated storage qubits
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)

        # Add the node to the graph and set "end_node" to True
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        G.nodes[node]["end_node"] = True

    # Repeat for each repeater node but set "end_node" to False
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

    # Add edges between nodes
    for node1, node2 in edges:
        G.add_edge("{}".format(node1), "{}".format(node2), capabilities=link_capability, weight=link_length)

    # Add edges between resources
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

    # Get link distance to capability data
    d_to_cap = load_link_data()

    # Networks have uniform link lengths so just graph the one set of capabilities
    link_capability = d_to_cap[str(link_length)]

    Gcq = nx.Graph()
    G = nx.Graph()

    # First make the center
    # Add communication and storage qubit nodes
    comm_qs = []
    storage_qs = []
    i = num_nodes - 1
    for c in range(num_rep_comm_q):
        comm_q_id = "{}-C{}".format(i, c)
        comm_qs.append(comm_q_id)
    for s in range(num_rep_storage_q):
        storage_q_id = "{}-S{}".format(i, s)
        storage_qs.append(storage_q_id)

    # Add nodes to graphs
    Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
    G.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs, end_node=False)

    # Then make the end nodes
    for i in range(num_nodes - 1):
        # Create communication and storage qubits
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)

        # Add them to graphs
        Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        G.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs, end_node=True)

        # Connect each end node to center
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
    # Get link distance to capability data
    d_to_cap = load_link_data()

    # Networks have uniform link lengths so just graph the one set of capabilities
    link_capability = d_to_cap[str(link_length)]

    Gcq = nx.Graph()
    G = nx.Graph()

    # Iterate over line and make each node end node
    for i in range(num_nodes):
        node = "{}".format(i)

        # Create communication and storage qubits
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)

        # Add to graphs
        Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs, end_node=True)

        # If internal node add edges
        if i > 0:
            prev_node_id = i - 1
            for j in range(num_end_node_comm_q):
                for k in range(num_end_node_storage_q):
                    Gcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(i, k))

            G.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability, weight=link_length)

    return Gcq, G


def create_surfnet_topology_gml(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3):
    # Load graph and get edges and nodes
    surfnet_graph = construct_graph(include='core')
    city_edges = list(set([tuple(sorted([e[0], e[1]])) for e in surfnet_graph.edges]))
    city_nodes = list(sorted(list(surfnet_graph.nodes)))
    city_node_to_id = dict([(node, str(2 * i)) for i, node in enumerate(city_nodes)])

    # Create edges for graph
    edges = [(city_node_to_id[node1], city_node_to_id[node2]) for node1, node2 in city_edges]
    repeater_nodes = list(city_node_to_id.values())
    end_nodes = []

    # Add extra nodes for each city
    for node in repeater_nodes:
        city_end_node = "{}".format(int(node) + 1)
        end_nodes.append(city_end_node)
        city_edge = (node, city_end_node)
        edges.append(city_edge)

    Gcq = nx.Graph()
    G = nx.Graph()

    # Add each end node to graph
    for node in end_nodes:
        # Create communication and storage qubits
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)

        # Add to graphs
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        G.nodes[node]["end_node"] = True

    # Create repeater nodes
    for node in repeater_nodes:
        # Create communication and storage qubits
        comm_qs = []
        storage_qs = []
        for c in range(num_rep_comm_q):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(num_rep_storage_q):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)

        # Add to graphs
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        G.nodes[node]["end_node"] = False

    # Connect all repeater nodes according to graph data
    lengths = []
    for node1, node2 in city_edges:
        if node1 in city_nodes and node2 in city_nodes:
            length = surfnet_graph.edges[(node1, node2)]['length']
            lengths.append(length)
            G.add_edge("{}".format(city_node_to_id[node1]), "{}".format(city_node_to_id[node2]), weight=link_length)

    for node in end_nodes:
        repeater_node = str(int(node) - 1)
        G.add_edge("{}".format(node), "{}".format(repeater_node), weight=link_length)

    # Add edges
    for node1, node2 in edges:
        num_comm_node1 = num_end_node_comm_q if G.nodes[node1]["end_node"] else num_rep_comm_q
        num_comm_node2 = num_end_node_comm_q if G.nodes[node2]["end_node"] else num_rep_comm_q
        for j in range(num_comm_node1):
            for k in range(num_comm_node2):
                Gcq.add_edge("{}-C{}".format(node1, j), "{}-C{}".format(node2, k))

    nx.write_gml(G, "Surfnet.gml")
    nx.write_gml(Gcq, "Surfnetcq.gml")

    return Gcq, G


def gen_surfnet_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3):
    """
    Generates a graph based on the SURFNET topology
    :param num_end_node_comm_q: type int
        Number of communication qubits each end node has
    :param num_end_node_storage_q: type int
        Number of storage qubits each end node has
    :param num_rep_comm_q: type int
        Number of communication qubits each repeater has
    :param num_rep_storage_q: type int
        Number of storage qubits each repeater has
    :return: type tuple
        Tuple of networkx.Graphs that represent the communication resources in the network and the connectivity of nodes
        in the network along with the link capabilities of each link.
    """
    # Link capability to use
    link_capability = [(0.999, 1400)]

    # Load GML files
    G = nx.read_gml("Surfnet.gml")
    Gcq = nx.read_gml("Surfnetcq.gml")

    for edge in G.edges:
        G.edges[edge]['capabilities'] = link_capability

    return Gcq, G


def gen_symm_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3,
                      link_length=5):
    # Get link distance to capability data
    d_to_cap = load_link_data()

    # Networks have uniform link lengths so just graph the one set of capabilities
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

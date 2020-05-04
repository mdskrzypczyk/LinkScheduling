import unittest
from jobscheduling.topology import gen_line_topology, gen_star_topology, gen_H_topology


class TestTopologies(unittest.TestCase):
    def test_line_topology(self):
        expected_num_comm = 1
        expected_num_stor = 3
        expected_num_nodes = 6
        expected_link_length = 5
        line_topology = gen_line_topology()

        communication_graph, connectivity_graph = line_topology
        self.assertEqual(len(connectivity_graph.nodes), expected_num_nodes)
        for node in connectivity_graph.nodes:
            self.assertEqual(len(connectivity_graph.nodes[node]["comm_qs"]), expected_num_comm)
            for i, comm_q in enumerate(connectivity_graph.nodes[node]["comm_qs"]):
                self.assertEqual(comm_q, "{}-C{}".format(node, i))
            self.assertEqual(len(connectivity_graph.nodes[node]["storage_qs"]), expected_num_stor)
            for i, stor_q in enumerate(connectivity_graph.nodes[node]["storage_qs"]):
                self.assertEqual(stor_q, "{}-S{}".format(node, i))

        self.assertEqual(len(connectivity_graph.edges), expected_num_nodes-1)
        for i, edge in enumerate(connectivity_graph.edges):
            self.assertEqual(edge, (str(i), str(i+1)))
            self.assertEqual(connectivity_graph.edges[edge]["weight"], expected_link_length)

        test_num_comm = 3
        test_num_stor = 5
        test_length = 15
        line_topology = gen_line_topology(num_end_node_comm_q=test_num_comm, num_end_node_storage_q=test_num_stor,
                                          link_length=test_length)

        communication_graph, connectivity_graph = line_topology
        self.assertEqual(len(connectivity_graph.nodes), expected_num_nodes)
        for node in connectivity_graph.nodes:
            self.assertEqual(len(connectivity_graph.nodes[node]["comm_qs"]), test_num_comm)
            for i, comm_q in enumerate(connectivity_graph.nodes[node]["comm_qs"]):
                self.assertEqual(comm_q, "{}-C{}".format(node, i))
            self.assertEqual(len(connectivity_graph.nodes[node]["storage_qs"]), test_num_stor)
            for i, stor_q in enumerate(connectivity_graph.nodes[node]["storage_qs"]):
                self.assertEqual(stor_q, "{}-S{}".format(node, i))

        self.assertEqual(len(connectivity_graph.edges), expected_num_nodes - 1)
        for i, edge in enumerate(connectivity_graph.edges):
            self.assertEqual(edge, (str(i), str(i + 1)))
            self.assertEqual(connectivity_graph.edges[edge]["weight"], test_length)


    def test_star_topology(self):
        expected_num_comm = 1
        expected_num_stor = 3
        expected_num_nodes = 5
        expected_link_length = 5
        star_topology = gen_star_topology()

        communication_graph, connectivity_graph = star_topology
        self.assertEqual(len(connectivity_graph.nodes), expected_num_nodes)
        for node in connectivity_graph.nodes:
            self.assertEqual(len(connectivity_graph.nodes[node]["comm_qs"]), expected_num_comm)
            for i, comm_q in enumerate(connectivity_graph.nodes[node]["comm_qs"]):
                self.assertEqual(comm_q, "{}-C{}".format(node, i))
            self.assertEqual(len(connectivity_graph.nodes[node]["storage_qs"]), expected_num_stor)
            for i, stor_q in enumerate(connectivity_graph.nodes[node]["storage_qs"]):
                self.assertEqual(stor_q, "{}-S{}".format(node, i))

        expected_edges = {
            ('4', '0'),
            ('4', '1'),
            ('4', '2'),
            ('4', '3')
        }
        self.assertEqual(len(connectivity_graph.edges), expected_num_nodes - 1)
        self.assertEqual(set(connectivity_graph.edges), expected_edges)
        for i, edge in enumerate(connectivity_graph.edges):
            self.assertEqual(connectivity_graph.edges[edge]["weight"], expected_link_length)

        test_num_comm = 3
        test_num_stor = 5
        test_length = 15
        star_topology = gen_star_topology(num_end_node_comm_q=test_num_comm, num_end_node_storage_q=test_num_stor,
                                          num_rep_comm_q=test_num_comm, num_rep_storage_q=test_num_stor,
                                          link_length=test_length)

        communication_graph, connectivity_graph = star_topology
        self.assertEqual(len(connectivity_graph.nodes), expected_num_nodes)
        for node in connectivity_graph.nodes:
            self.assertEqual(len(connectivity_graph.nodes[node]["comm_qs"]), test_num_comm)
            for i, comm_q in enumerate(connectivity_graph.nodes[node]["comm_qs"]):
                self.assertEqual(comm_q, "{}-C{}".format(node, i))
            self.assertEqual(len(connectivity_graph.nodes[node]["storage_qs"]), test_num_stor)
            for i, stor_q in enumerate(connectivity_graph.nodes[node]["storage_qs"]):
                self.assertEqual(stor_q, "{}-S{}".format(node, i))

        expected_edges = {
            ('4', '0'),
            ('4', '1'),
            ('4', '2'),
            ('4', '3')
        }
        self.assertEqual(len(connectivity_graph.edges), expected_num_nodes - 1)
        self.assertEqual(set(connectivity_graph.edges), expected_edges)
        for i, edge in enumerate(connectivity_graph.edges):
            self.assertEqual(connectivity_graph.edges[edge]["weight"], test_length)

    def test_H_topology(self):
        expected_num_comm = 1
        expected_num_stor = 3
        expected_num_nodes = 6
        expected_link_length = 5
        h_topology = gen_H_topology()

        communication_graph, connectivity_graph = h_topology
        self.assertEqual(len(connectivity_graph.nodes), expected_num_nodes)
        for node in connectivity_graph.nodes:
            self.assertEqual(len(connectivity_graph.nodes[node]["comm_qs"]), expected_num_comm)
            for i, comm_q in enumerate(connectivity_graph.nodes[node]["comm_qs"]):
                self.assertEqual(comm_q, "{}-C{}".format(node, i))
            self.assertEqual(len(connectivity_graph.nodes[node]["storage_qs"]), expected_num_stor)
            for i, stor_q in enumerate(connectivity_graph.nodes[node]["storage_qs"]):
                self.assertEqual(stor_q, "{}-S{}".format(node, i))

        expected_edges = {
            ('0', '1'),
            ('2', '1'),
            ('1', '4'),
            ('3', '4'),
            ('5', '4')
        }
        self.assertEqual(len(connectivity_graph.edges), expected_num_nodes - 1)
        self.assertEqual(set(connectivity_graph.edges), expected_edges)
        for i, edge in enumerate(connectivity_graph.edges):
            self.assertEqual(connectivity_graph.edges[edge]["weight"], expected_link_length)

        test_num_comm = 3
        test_num_stor = 5
        test_length = 15
        h_topology = gen_H_topology(num_end_node_comm_q=test_num_comm, num_end_node_storage_q=test_num_stor,
                                          num_rep_comm_q=test_num_comm, num_rep_storage_q=test_num_stor,
                                          link_length=test_length)

        communication_graph, connectivity_graph = h_topology
        self.assertEqual(len(connectivity_graph.nodes), expected_num_nodes)
        for node in connectivity_graph.nodes:
            self.assertEqual(len(connectivity_graph.nodes[node]["comm_qs"]), test_num_comm)
            for i, comm_q in enumerate(connectivity_graph.nodes[node]["comm_qs"]):
                self.assertEqual(comm_q, "{}-C{}".format(node, i))
            self.assertEqual(len(connectivity_graph.nodes[node]["storage_qs"]), test_num_stor)
            for i, stor_q in enumerate(connectivity_graph.nodes[node]["storage_qs"]):
                self.assertEqual(stor_q, "{}-S{}".format(node, i))

        expected_edges = {
            ('0', '1'),
            ('2', '1'),
            ('1', '4'),
            ('3', '4'),
            ('5', '4')
        }
        self.assertEqual(len(connectivity_graph.edges), expected_num_nodes - 1)
        self.assertEqual(set(connectivity_graph.edges), expected_edges)
        for i, edge in enumerate(connectivity_graph.edges):
            self.assertEqual(connectivity_graph.edges[edge]["weight"], test_length)

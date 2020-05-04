import unittest
from jobscheduling.protocolgen import Protocol, LinkProtocol, SwapProtocol, DistillationProtocol, create_protocol, \
    esss, find_pumping_protocol, find_binary_protocol, get_protocol_for_link
from jobscheduling.topology import gen_line_topology


class TestProtocolClasses(unittest.TestCase):
    def test_protocol(self):
        test_fidelity = 0.8
        test_rate = 1.5
        test_nodes = ['1', '2']
        protocol = Protocol(F=test_fidelity, R=test_rate, nodes=test_nodes)

        self.assertEqual(protocol.F, test_fidelity)
        self.assertEqual(protocol.R, test_rate)
        self.assertEqual(protocol.duration, 1 / test_rate)
        self.assertEqual(protocol.nodes, test_nodes)

    def test_link_protocol(self):
        test_fidelity = 0.8
        test_rate = 1.5
        test_nodes = ['1', '2']
        protocol = LinkProtocol(F=test_fidelity, R=test_rate, nodes=test_nodes)

        self.assertEqual(protocol.F, test_fidelity)
        self.assertEqual(protocol.R, test_rate)
        self.assertEqual(protocol.duration, 1 / test_rate)
        self.assertEqual(protocol.dist, protocol.duration)
        self.assertEqual(protocol.nodes, test_nodes)

    def test_swap_protocol(self):
        test_link_fidelity = 0.9
        test_link_rate = 1
        test_link_nodes = ['1', '2']
        test_link_fidelity2 = 0.85
        test_link_rate2 = 2
        test_link_nodes2 = ['2', '3']
        test_protocols = [LinkProtocol(F=test_link_fidelity, R=test_link_rate, nodes=test_link_nodes),
                          LinkProtocol(F=test_link_fidelity2, R=test_link_rate2, nodes=test_link_nodes2)]

        test_fidelity = 0.8
        test_rate = 1.5
        test_nodes = ['1']
        protocol = SwapProtocol(F=test_fidelity, R=test_rate, protocols=test_protocols, nodes=test_nodes)

        self.assertEqual(protocol.F, test_fidelity)
        self.assertEqual(protocol.R, test_rate)
        self.assertEqual(protocol.duration, 0.01)
        self.assertEqual(protocol.dist, test_protocols[0].duration + protocol.duration)
        self.assertEqual(protocol.nodes, test_nodes)
        self.assertEqual(protocol.protocols, test_protocols)

    def test_distill_protocol(self):
        test_link_fidelity = 0.9
        test_link_rate = 1
        test_nodes = ['1', '2']
        test_link_fidelity2 = 0.85
        test_link_rate2 = 2
        test_protocols = [LinkProtocol(F=test_link_fidelity, R=test_link_rate, nodes=test_nodes),
                          LinkProtocol(F=test_link_fidelity2, R=test_link_rate2, nodes=test_nodes)]

        test_fidelity = 0.8
        test_rate = 1.5
        protocol = DistillationProtocol(F=test_fidelity, R=test_rate, protocols=test_protocols, nodes=test_nodes)

        self.assertEqual(protocol.F, test_fidelity)
        self.assertEqual(protocol.R, test_rate)
        self.assertEqual(protocol.duration, 0.01)
        self.assertEqual(protocol.dist, test_protocols[0].duration + protocol.duration)
        self.assertEqual(protocol.nodes, test_nodes)
        self.assertEqual(protocol.protocols, test_protocols)


class TestProtocolGeneration(unittest.TestCase):
    def test_create_protocol_link(self):
        _, G = gen_line_topology()
        test_source = '0'
        test_dest = '1'
        test_fidelity = 0.8
        test_rate = 0.1

        path = [test_source, test_dest]
        protocol = create_protocol(path, G, test_fidelity, test_rate)
        self.assertEqual(type(protocol), LinkProtocol)
        self.assertEqual(protocol.nodes, path)
        self.assertEqual(protocol.F, 0.8338445929403213)
        self.assertEqual(protocol.R, 20.837302311286685)

    def test_create_swap_protocol_hop(self):
        _, G = gen_line_topology()
        test_source = '0'
        repeater = '1'
        test_dest = '2'
        test_fidelity = 0.7
        test_rate = 0.1

        path = [test_source, repeater, test_dest]
        protocol = create_protocol(path, G, test_fidelity, test_rate)
        self.assertEqual(type(protocol), SwapProtocol)
        self.assertEqual(protocol.nodes, [repeater])
        self.assertEqual(protocol.F, 0.7044993449408659)
        self.assertEqual(protocol.R, 9.090909090909092)

        protocolLeft, protocolRight = protocol.protocols
        self.assertEqual(type(protocolLeft), LinkProtocol)
        self.assertEqual(protocolLeft.nodes, [test_source, repeater])
        self.assertEqual(protocolLeft.F, 0.8338445929403213)
        self.assertEqual(protocolLeft.R, 20.837302311286685)

        self.assertEqual(type(protocolRight), LinkProtocol)
        self.assertEqual(protocolRight.nodes, [repeater, test_dest])
        self.assertEqual(protocolRight.F, 0.8338445929403213)
        self.assertEqual(protocolRight.R, 20.837302311286685)

    def test_create_distillation_protocol_hop(self):
        _, G = gen_line_topology()
        test_source = '0'
        repeater = '1'
        test_dest = '2'
        test_fidelity = 0.8
        test_rate = 0.1

        path = [test_source, repeater, test_dest]
        protocol = create_protocol(path, G, test_fidelity, test_rate)
        self.assertEqual(type(protocol), DistillationProtocol)
        self.assertEqual(protocol.nodes, [test_source, test_dest])
        self.assertEqual(protocol.F, 0.8130731716322149)
        self.assertEqual(protocol.R, 3.0303030303030303)

        swapLeft, swapRight = protocol.protocols
        self.assertEqual(type(swapLeft), SwapProtocol)
        self.assertEqual(swapLeft.nodes, [repeater])
        self.assertEqual(swapLeft.F, 0.7743109571155198)
        self.assertEqual(swapLeft.R, 14.15833266978285)

        self.assertEqual(type(swapRight), SwapProtocol)
        self.assertEqual(swapRight.nodes, [repeater])
        self.assertEqual(swapRight.F, 0.7743109571155198)
        self.assertEqual(swapRight.R, 14.15833266978285)

        protocolLeft, protocolRight = swapLeft.protocols
        self.assertEqual(type(protocolLeft), LinkProtocol)
        self.assertEqual(protocolLeft.nodes, [test_source, repeater])
        self.assertEqual(protocolLeft.F, 0.8770831028154402)
        self.assertEqual(protocolLeft.R, 14.15833266978285)

        self.assertEqual(type(protocolRight), LinkProtocol)
        self.assertEqual(protocolRight.nodes, [repeater, test_dest])
        self.assertEqual(protocolRight.F, 0.8770831028154402)
        self.assertEqual(protocolRight.R, 14.15833266978285)

        protocolLeft, protocolRight = swapRight.protocols
        self.assertEqual(type(protocolLeft), LinkProtocol)
        self.assertEqual(protocolLeft.nodes, [test_source, repeater])
        self.assertEqual(protocolLeft.F, 0.8770831028154402)
        self.assertEqual(protocolLeft.R, 14.15833266978285)

        self.assertEqual(type(protocolRight), LinkProtocol)
        self.assertEqual(protocolRight.nodes, [repeater, test_dest])
        self.assertEqual(protocolRight.F, 0.8770831028154402)
        self.assertEqual(protocolRight.R, 14.15833266978285)

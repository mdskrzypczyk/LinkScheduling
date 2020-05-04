import networkx as nx
from math import ceil
from copy import copy
from jobscheduling.log import LSLogger
from jobscheduling.protocols import get_protocol_rate
from jobscheduling.qmath import swap_links, unswap_links, distill_links, undistill_link_even, \
    fidelity_for_distillations, distillations_for_fidelity

logger = LSLogger()


class Protocol:
    def __init__(self, F, R, nodes):
        """
        Basic class for a protocol action
        :param F: type float
            Fidelity of the protocol
        :param R: type float
            Rate at which protocol may be executed
        :param nodes: type list
            Set of nodes involved in protocol action
        """
        self.F = F
        self.R = R
        self.nodes = nodes
        self.set_duration(R)
        self.dist = 0

    def set_duration(self, R):
        if R == 0:
            self.duration = float('inf')
        else:
            self.duration = 1 / R


class LinkProtocol(Protocol):
    name_template = "LG{};{};{}"
    count = 0

    def __init__(self, F, R, nodes):
        """
        Protocol action class for Link generation
        :param F: type float
            The initial fidelity of the generated link
        :param R: type float
            The rate at which the link may be generated
        :param nodes: type list
            List of the nodes involved in generating the link
        """
        super(LinkProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.name = self.name_template.format(self.count, *nodes)
        self.dist = self.duration
        LinkProtocol.count += 1

    def __copy__(self):
        return LinkProtocol(F=self.F, R=self.R, nodes=self.nodes)


class DistillationProtocol(LinkProtocol):
    name_template = "D{};{};{}"
    count = 0
    distillation_duration = 0.01

    def __init__(self, F, R, protocols, nodes):
        """
        Protocol action class for entanglement distillation
        :param F: type float
            The fidelity of the link after distillation
        :param R: type float
            The rate at which distillation may be performed
        :param protocols: type list
            List of the two protocol actions the distillation builds upon
        :param nodes: type list
            List of the nodes performing entanglement distillation
        """
        super(DistillationProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.protocols = list(sorted(protocols, key=lambda p: p.R))
        self.durations = [protocol.duration for protocol in self.protocols]
        self.dist = max([protocol.dist for protocol in self.protocols]) + self.duration
        self.name = self.name_template.format(self.count, *nodes)
        DistillationProtocol.count += 1

    def set_duration(self, R):
        self.duration = self.distillation_duration

    def __copy__(self):
        return DistillationProtocol(F=self.F, R=self.R, nodes=self.nodes, protocols=[copy(p) for p in self.protocols])


class SwapProtocol(Protocol):
    name_template = "S{};{}"
    count = 0
    swap_duration = 0.01

    def __init__(self, F, R, protocols, nodes):
        """
        Protocol action class for performing entanglement swapping
        :param F: type float
            Fidelity of the link after performing entanglement swapping
        :param R: type float
            Rate at which entanglement swapping may be performed
        :param protocols: type list
            List of the protocol actions the swap builds upon
        :param nodes: type list
            List of the nodes that perform the entanglement swap
        """
        super(SwapProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.protocols = list(sorted(protocols, key=lambda p: p.R))
        self.durations = [protocol.duration for protocol in self.protocols]
        self.dist = max([protocol.dist for protocol in self.protocols]) + self.duration
        self.name = self.name_template.format(self.count, *nodes)
        SwapProtocol.count += 1

    def set_duration(self, R):
        self.duration = self.swap_duration

    def __copy__(self):
        return SwapProtocol(F=self.F, R=self.R, nodes=self.nodes, protocols=[copy(p) for p in self.protocols])


cache = {}


def create_protocol(path, nodeG, Fmin, Rmin):
    """
    Attempts to create a protocol over the repeater chain specified by the path in the graph nodeG. If a protocol is
    found it satisfies the minimum fidelity Fmin and rate Rmin
    :param path: type list
        List of the nodes that form the repeater chain in nodeG
    :param nodeG: type ~networkx.Graph
        Graph representing the quantum network
    :param Fmin: type float
        The minimum fidelity the protocol must provide
    :param Rmin: type float
        The minimum rate at which the protocol may be executed
    :return: type Protocol
        The sink protocol action of the repeater protocol for the path
    """
    def filter_node(node):
        return node in path

    def filter_edge(node1, node2):
        return node1 in path and node2 in path

    logger.debug("Creating protocol on path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
    subG = nx.subgraph_view(nodeG, filter_node, filter_edge)

    pathResources = {}
    cache_key = [Fmin, Rmin]
    cache_key += [node for node in sorted(path)]
    for node in sorted(path):
        numCommResources = len(nodeG.nodes[node]['comm_qs'])
        numStorResources = len(nodeG.nodes[node]['storage_qs'])
        pathResources[node] = {
            "comm": numCommResources,
            "storage": numStorResources,
            "total": numCommResources + numStorResources
        }
        cache_key.append(numCommResources)
        cache_key.append(numStorResources)

    cache_key = tuple(cache_key)
    # if cache_key in cache.keys():
    #     return cache[cache_key]

    try:
        protocol = esss(path, pathResources, subG, Fmin, Rmin)
        if type(protocol) != Protocol and protocol is not None:
            cache[cache_key] = protocol
            return protocol
        else:
            cache[cache_key] = None
            return None
    except Exception:
        logger.exception("Failed to create protocol for path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
        raise Exception()


def esss(path, pathResources, G, Fmin, Rmin):
    """
    Performs entanglement swap scheme search (ESSS) to find a protocol connecting the end nodes of path in the graph G.
    :param path: type list
        List of nodes that form the path to find a protocol for
    :param pathResources: type dict
        Dictionary of node to resources that are available at the node for generating entanglement
    :param G: type networkx.Graph
        Graph representing the quantum network
    :param Fmin: type float
        The minimum fidelity desired from the protocol
    :param Rmin: type float
        The minimum rate desired from the protocol
    :return: type Protocol
        The sink protocol action for the repeater protocol to use over the path
    """
    logger.debug("ESSS on path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
    if len(path) == 2:
        logger.debug("Down to elementary link, finding distillation protocol")
        link_properties = G.get_edge_data(*path)['capabilities']
        protocol = get_protocol_for_link(link_properties, G, Fmin, Rmin, path, pathResources)
        return protocol

    else:
        # Start by dividing the path in two
        lower = 1
        upper = len(path) - 1
        protocols = []
        while lower < upper:
            numL = (upper + lower + 1) // 2
            numR = len(path) + 1 - numL

            logger.debug("Finding protocol for path {} with pivot {}".format(path, path[numL - 1]))
            possible_protocol, Rl, Rr = find_split_path_protocol(path, dict(pathResources), G, Fmin, Rmin, numL, numR)

            if possible_protocol and possible_protocol.F >= Fmin and possible_protocol.R >= Rmin:
                protocols.append(possible_protocol)

            if Rl == Rr:
                logger.debug("Rates on left and right paths balanced")
                break

            elif Rl < Rr:
                logger.debug("Rate on left path lower, extending right path")
                upper -= 1
            else:
                logger.debug("Rate on right path lower, extending left path")
                lower += 1

        protocol = list(sorted(protocols, key=lambda p: (p.R, p.F)))[-1] if protocols else None
        if protocol is None:
            logger.debug("Failed to find protocol for path {} achieving Fmin {} and Rmin {}".format(path, Fmin, Rmin))
        else:
            rate = get_protocol_rate(('', '', Fmin, Rmin), protocol, (None, G))
            protocol.R = rate

        return protocol


def find_split_path_protocol(path, pathResources, G, Fmin, Rmin, numL, numR):
    """
    Finds a protocol by splitting the path into two sub-paths and constructing protocols for the sub-paths. A protocol
    for the full path is then constructed by performing a swap at the pivot node specified by numL
    :param path: type list
        List of nodes that form the repeater chain
    :param pathResources: type dict
        Dictionary of node to resources for generating entanglement
    :param G: type networkx.Graph
        A graph representing the quantum network
    :param Fmin: type float
        The minimum desired fidelity of the protocol
    :param Rmin: type float
        The minimum desired rate of the protocol
    :param numL: type int
        Number of nodes in the left sub-path
    :param numR: type int
        Number of nodes in the right sub-path
    :return: type Protocol
        The sink node protocol action for the protocol to execute over the path
    """
    protocols = []

    resourceCopy = dict([(k, dict(v)) for k, v in pathResources.items()])

    # If we are swapping the middle node needs to use one resource to hold an end of the first link
    resourceCopy[path[numL - 1]]['total'] -= 1
    resourceCopy[path[numL - 1]]['storage'] -= 1

    # Assume we allocate half the comm resources of pivot node to either link
    num = 0
    while True:
        # Compute minimum fidelity in order for num distillations to achieve Fmin
        Fminswap = fidelity_for_distillations(num, Fmin)
        if Fminswap == 0:
            break

        Funswapped = unswap_links(Fminswap)

        # Calculate the needed rates of the links
        if num > 0:
            Rlink = Rmin * (num + 1) / resourceCopy[path[numL - 1]]['comm'] / 2
        else:
            Rlink = Rmin

        pathResourcesCopy = dict([(k, dict(v)) for k, v in resourceCopy.items()])

        # If we are distilling then the end nodes need to hold one link between protocol steps
        if num > 0:
            pathResourcesCopy[path[0]]['storage'] -= 1
            pathResourcesCopy[path[-1]]['storage'] -= 1
            pathResourcesCopy[path[0]]['total'] -= 1
            pathResourcesCopy[path[-1]]['total'] -= 1

        if pathResourcesCopy[path[0]]['storage'] < 0 or pathResourcesCopy[path[-1]]['storage'] < 0:
            return None, 0, 0

        # Search for protocols on left and right that have above properties
        protocolL = esss(path[:numL], pathResourcesCopy, G, Funswapped, Rlink)
        protocolR = esss(path[-numR:], pathResourcesCopy, G, Funswapped, Rlink)

        # Add to list of protocols
        if protocolL is not None and protocolR is not None and type(protocolL) != Protocol and \
                type(protocolR) != Protocol:
            Fswap = swap_links(protocolL.F, protocolR.F)
            Rswap = min(protocolL.R, protocolR.R)
            swap_protocol = SwapProtocol(F=Fswap, R=Rswap, protocols=[protocolL, protocolR], nodes=[path[-numR]])
            protocol = copy(swap_protocol)
            for i in range(num):
                Fdistilled = distill_links(protocol.F, Fswap)
                Rdistilled = Rswap / (i + 2)
                protocol = DistillationProtocol(F=Fdistilled, R=Rdistilled, protocols=[protocol, copy(swap_protocol)],
                                                nodes=[path[0], path[-1]])

            logger.debug("Found Swap/Distill protocol achieving F={},R={},numSwappedDistills={}".format(protocol.F,
                                                                                                        protocol.R,
                                                                                                        num))
            logger.debug("Underlying link protocols have Fl={},Rl={} and Fr={},Rr={}".format(protocolL.F, protocolL.R,
                                                                                             protocolR.F, protocolR.R))

            protocols.append((protocol, protocolL.R, protocolR.R))
        num += 1

    # Choose protocol with maximum rate > Rmin
    if protocols:
        if len(path) <= 3:
            protocols = sorted(protocols, key=lambda p: (p[0].R, p[0].F))[-5:]
            for protocol, Rl, Rr in protocols:
                if protocol is not None:
                    rate = get_protocol_rate(('', '', Fmin, Rmin), protocol, (None, G))
                    protocol.R = rate
        protocol, Rl, Rr = sorted(protocols, key=lambda p: (p[0].R, p[0].F))[-1]
        logger.debug("Found Swap/Distill protocol achieving F={},R={},numSwappedDistills={}".format(protocol.F,
                                                                                                    protocol.R,
                                                                                                    num + 1))

        return protocol, Rl, Rr

    else:
        logger.debug("Failed to find protocol for path {} achieving Fmin {} and Rmin {}".format(path, Fmin, Rmin))
        return None, 0, 0


def get_protocol_for_link(link_properties, G, Fmin, Rmin, nodes, nodeResources):
    """
    Finds a protocol for generating entanglement between two directly connected nodes
    :param link_properties: type list
        List of (fidelity, rate) capabilities that the link between the nodes supports
    :param G: type networkx.Graph
        Graph representing the quantum network
    :param Fmin: type float
        The minimum desired fidelity from the protocol
    :param Rmin: type float
        The minimum desired rate from the protocol
    :param nodes: type list
        A list of the two nodes that are connected in G to find a protocol for
    :param nodeResources: type dict
        A dictionary of node to resources for generating entanglement
    :return: type Protocol
        The sink node protocol action of the repeater protocol to use between the two nodes
    """
    logger.debug("Getting protocol on link {} with Fmin {} and Rmin {}".format(nodes, Fmin, Rmin))
    if all([R < Rmin for _, R in link_properties]):
        logger.debug("Cannot satisfy rate {} between nodes {}".format(Rmin, nodes))
        return None

    if any([v < 1 for v in [nodeResources[n]['total'] for n in nodes]]):
        logger.debug("Not enough resources to generate link between nodes {}".format(nodes))
        return None

    # Check if any single link generation protocols exist
    protocols = []
    for F, R in link_properties:
        if R >= Rmin and F >= Fmin:
            protocols.append(LinkProtocol(F=F, R=R, nodes=nodes))

    if protocols:
        return list(sorted(protocols, key=lambda p: (p.R, p.F)))[-1]

    # logger.debug("Link not capable of generating without distillation")
    if any([v < 2 for v in [nodeResources[n]['total'] for n in nodes]]):
        logger.debug("Not enough resources to perform distillation with nodes {}".format(nodes))
        return None

    # Search for the link gen protocol with highest rate that satisfies fidelity
    minResources = min([nodeResources[n]["total"] for n in nodes])
    pumping_protocol = find_pumping_protocol(nodes, nodeResources, G, link_properties, Fmin, Rmin)
    if minResources > 2:
        binary_protocol = find_binary_protocol(nodes, nodeResources, G, link_properties, Fmin, Rmin)
        if binary_protocol:
            if not pumping_protocol or binary_protocol.R > pumping_protocol.R:
                return binary_protocol

    return pumping_protocol


def find_pumping_protocol(nodes, nodeResources, G, link_properties, Fmin, Rmin):
    """
    Finds an entanglement pumping protocol between two directly connected nodes
    :param nodes: type list
        List of the nodes to find the protocol for
    :param nodeResources: type dict
        Dictionary of node to resources for generating entanglement
    :param G: type networkx.Graph
        Graph representing the quantum network
    :param link_properties: type list
        List of (fidelity,rate) capabilities that the link between the nodes supports
    :param Fmin: type float
        The minimum desired fidelity from the pumping protocol
    :param Rmin: type float
        The minimum desired rate from the pumping protocol
    :return: type Protocol
        The sink node protocol action for the repeater protocol to use between the nodes
    """
    # Search for the link gen protocol with highest rate that satisfies fidelity
    protocols = []
    # Can only generate as fast as the most constrained node
    minNodeComms = min([nodeResources[n]['comm'] for n in nodes])
    for F, R in link_properties:
        if R < Rmin:
            continue
        currF = F
        currProtocol = LinkProtocol(F=F, R=R, nodes=nodes)
        numGens = distillations_for_fidelity(F, Fmin)

        if numGens != float('inf'):
            generationLatency = 1 / (R / ceil(numGens / minNodeComms))
            distillLatency = numGens * DistillationProtocol.distillation_duration
            currR = 1 / (generationLatency + distillLatency)
            linkProtocol = LinkProtocol(F=F, R=R, nodes=nodes)
            for i in range(numGens):
                currF = distill_links(currF, F)
                currProtocol = DistillationProtocol(F=currF, R=currR, protocols=[currProtocol, linkProtocol],
                                                    nodes=nodes)

            if currProtocol.F > Fmin and currProtocol.R >= Rmin:
                logger.debug(
                    "Found distillation protocol using F={},R={},numGens={}".format(currProtocol.F, currProtocol.R,
                                                                                    numGens))
                protocols.append(currProtocol)

    protocol = list(sorted(protocols, key=lambda p: (p.R, p.F)))[-1] if protocols else None
    if protocol is None:
        logger.debug("Failed to find protocol for path {} achieving Fmin {} and Rmin {}".format(nodes, Fmin, Rmin))
    else:
        rate = get_protocol_rate(('', '', Fmin, Rmin), protocol, (None, G))
        protocol.R = rate

    return protocol


def find_binary_protocol(nodes, nodeResources, G, link_properties, Fmin, Rmin):
    """
    Finds a nested entanglement distillation protocol
    :param nodes: type list
        List of the connected nodes to create the protocol for
    :param nodeResources: type dict
        Dictionary of node to resources for entanglement generation
    :param G: type networkx.Graph
        Graph representing the quantum network
    :param link_properties: type list
        List of (fidelity, rate) capabilities that the link supports
    :param Fmin: type float
        The minimum desired fidelity of the distillation protocol
    :param Rmin: type float
        The minimum desired rate of the distillation protocol
    :return: type Protocol
        The sink node protocol action of the protocol to use between the nodes
    """
    # Search for the link gen protocol with highest rate that satisfies fidelity
    protocols = []
    # Can only generate as fast as the most constrained node
    minNodeComms = min([nodeResources[n]['comm'] for n in nodes])
    minResources = min([nodeResources[n]["total"] for n in nodes])

    for F, R in link_properties:
        if R < Rmin:
            continue
        numDist = distillations_for_const_fidelity(F, Fmin)
        numGens = links_for_distillations(numDist)

        # Check that numGens is below the max supported by minResources
        if numGens > 2**(minResources - 1):
            continue

        numRounds = num_rounds(numGens, minNodeComms, minResources - minNodeComms)
        binary_rate = R / numRounds
        if binary_rate < Rmin:
            continue

        q = [LinkProtocol(F=F, R=R, nodes=nodes) for _ in range(numGens)]
        currProtocol = None
        while len(q) > 1:
            p1 = q.pop(0)
            p2 = q.pop(0)
            currF = distill_links(p1.F, p2.F)
            currProtocol = DistillationProtocol(F=currF, R=min(p1.R, p2.R), protocols=[p1, p2], nodes=nodes)
            q.append(currProtocol)

        currProtocol = q.pop(0)
        currProtocol.R = binary_rate
        if currProtocol.F > Fmin and currProtocol.R >= Rmin:
            logger.debug(
                "Found distillation protocol using F={},R={},numGens={}".format(currProtocol.F, currProtocol.R,
                                                                                numGens))
            protocols.append(currProtocol)

    protocol = list(sorted(protocols, key=lambda p: (p.R, p.F)))[-1] if protocols else None
    if protocol is None:
        logger.debug("Failed to find protocol for path {} achieving Fmin {} and Rmin {}".format(nodes, Fmin, Rmin))
    else:
        rate = get_protocol_rate(('', '', Fmin, Rmin), protocol, (None, G))
        protocol.R = rate

    return protocol


def const_fidelity_distillation_max(Finitial, num):
    """
    Finds the fidelity of the final link when performing num levels of nested entanglement distillation using an initial
    link fidelity of Finitial.
    :param Finitial: type float
        The initial link fidelity
    :param num: type int
        The number of levels of nested entanglement distillation
    :return: type float
        The maximum fidelity achieved
    """
    Fout = Finitial
    for i in range(num):
        Fout = distill_links(Fout, Fout)

    return Fout


def distillations_for_const_fidelity(Finitial, Ftarget):
    """
    Finds the number of distillations needed for achieving Ftarget using nested entanglement distillation with an
    initial link fidelity of Finitial
    :param Finitial: type float
        The initial fidelity of the link
    :param Ftarget: type float
        The target fidelity of the link
    :return: type int
        The number of distillations performed in the nested entanglement distillation achieving Ftarget
    """
    if Finitial < 0.5:
        return float('inf')
    num = 0
    while Ftarget > Finitial:
        num += 1
        Ftarget = undistill_link_even(Ftarget)

    return 2**num - 1


def links_for_distillations(num_distillations):
    """
    Returns the number of links needed to perform num_distillations distillation operations
    :param num_distillations: type int
        The number of distillations to perform
    :return: type int
        The number of links required
    """
    return num_distillations + 1


def bitcount(n):
    count = 0
    while n > 0:
        count = count + 1
        n = n & (n - 1)
    return count


def num_rounds(num_links, num_comm, num_storage):
    """
    Finds the number of rounds of entanglement generation needed to generate num_links links using num_comm
    communication qubits and num_storage storage qubits
    :param num_links: type int
        The number of links to be generated
    :param num_comm: type int
        The number of communication qubits available
    :param num_storage: type int
        The number of storage qubits available
    :return: type int
        The number of rounds of generating entanglement needed to generate num_links links
    """
    num_rounds = 0
    generated = 0
    while generated < num_links:
        occupied_resources = bitcount(generated)
        num_available = min(num_comm, num_comm + num_storage - occupied_resources)
        num_rounds += 1
        generated += num_available

    return num_rounds

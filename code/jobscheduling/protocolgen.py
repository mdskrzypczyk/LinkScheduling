import networkx as nx
from math import ceil
from copy import copy
from jobscheduling.log import LSLogger
from jobscheduling.qmath import swap_links, unswap_links, distill_links, fidelity_for_distillations, distillations_for_fidelity


logger = LSLogger()


class Protocol:
    def __init__(self, F, R, nodes):
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


def create_protocol(path, nodeG, Fmin, Rmin):
    def filter_node(node):
        return node in path

    def filter_edge(node1, node2):
        return node1 in path and node2 in path

    logger.debug("Creating protocol on path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
    subG = nx.subgraph_view(nodeG, filter_node, filter_edge)

    pathResources = {}
    for node in path:
        numCommResources = len(nodeG.nodes[node]['comm_qs'])
        numStorResources = len(nodeG.nodes[node]['storage_qs'])
        pathResources[node] = {
            "comm": numCommResources,
            "total": numCommResources + numStorResources
        }

    try:
        protocol = esss(path, pathResources, subG, Fmin, Rmin)
        if type(protocol) != Protocol and protocol is not None:
            return protocol
        else:
            return None
    except:
        logger.exception("Failed to create protocol for path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
        raise Exception()


def esss(path, pathResources, G, Fmin, Rmin):
    logger.debug("ESSS on path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
    if len(path) == 2:
        logger.debug("Down to elementary link, finding distillation protocol")
        link_properties = G.get_edge_data(*path)['capabilities']
        protocol = get_protocol_for_link(link_properties, Fmin, Rmin, path, pathResources)
        return protocol

    else:
        # Start by dividing the path in two
        lower = 1
        upper = len(path) - 1
        protocol = None
        rate = 0
        fidelity = 0
        while lower < upper:
            numL = (upper + lower + 1) // 2
            numR = len(path) + 1 - numL

            logger.debug("Finding protocol for path {} with pivot {}".format(path, path[numL]))
            possible_protocol, Rl, Rr = find_split_path_protocol(path, pathResources, G, Fmin, Rmin, numL, numR)

            if Rl == Rr:
                logger.debug("Rates on left and right paths balanced")
                return possible_protocol
            elif Rl < Rr:
                logger.debug("Rate on left path lower, extending right path")
                upper -= 1
            else:
                logger.debug("Rate on right path lower, extending left path")
                lower += 1

            if possible_protocol and possible_protocol.F >= Fmin and possible_protocol.R >= Rmin:
                if possible_protocol.R >= rate and possible_protocol.F >= fidelity:
                    protocol = possible_protocol
                    rate = protocol.R
                    fidelity = protocol.F

        return protocol


def find_split_path_protocol(path, pathResources, G, Fmin, Rmin, numL, numR):
    protocols = []

    resourceCopy = dict(pathResources)

    # If we are swapping the middle node needs to use one resource to hold an end of the first link
    resourceCopy[path[numL - 1]]['total'] -= 1

    # Assume we allocate half the comm resources of pivot node to either link
    # resourceCopy[path[numL - 1]]['comm'] = max(resourceCopy[path[numL - 1]]['comm'] // 2, 1)
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

        pathResourcesCopy = dict(resourceCopy)

        # If we are distilling then the end nodes need to hold one link between protocol steps
        if num > 0:
            pathResourcesCopy[path[0]]['total'] -= 1
            pathResourcesCopy[path[-1]]['total'] -= 1

        # Search for protocols on left and right that have above properties
        protocolL = esss(path[:numL], pathResourcesCopy, G, Funswapped, Rlink)
        protocolR = esss(path[-numR:], pathResourcesCopy, G, Funswapped, Rlink)

        # Add to list of protocols
        if protocolL is not None and protocolR is not None and type(protocolL) != Protocol and type(protocolR) != Protocol:
            Fswap = swap_links(protocolL.F, protocolR.F)
            Rswap = min(protocolL.R, protocolR.R)
            swap_protocol = SwapProtocol(F=Fswap, R=Rswap, protocols=[protocolL, protocolR], nodes=[path[-numR]])
            protocol = copy(swap_protocol)
            for i in range(num):
                Fdistilled = distill_links(protocol.F, Fswap)
                Rdistilled = Rswap / (i + 2)
                protocol = DistillationProtocol(F=Fdistilled, R=Rdistilled, protocols=[protocol, copy(swap_protocol)],
                                                nodes=[path[0], path[-1]])

            logger.debug("Found Swap/Distill protocol achieving F={},R={},numSwappedDistills={}".format(protocol.F, protocol.R,
                                                                                                 num))
            logger.debug("Underlying link protocols have Fl={},Rl={} and Fr={},Rr={}".format(protocolL.F, protocolL.R,
                                                                                      protocolR.F, protocolR.R))
            protocols.append((protocol, protocolL.R, protocolR.R))
            num += 1

        else:
            Rl = 0 if not protocolL else protocolL.R
            Rr = 0 if not protocolR else protocolR.R
            protocols.append((Protocol(F=0, R=0, nodes=None), Rl, Rr))
            break

    # Choose protocol with maximum rate > Rmin
    if protocols:
        protocol, Rl, Rr = sorted(protocols, key=lambda p: p[0].R)[-1]
        logger.debug("Found Swap/Distill protocol achieving F={},R={},numSwappedDistills={}".format(protocol.F, protocol.R,
                                                                                             num + 1))

        return protocol, Rl, Rr

    else:
        return None, 0, 0


def get_protocol_for_link(link_properties, Fmin, Rmin, nodes, nodeResources):
    logger.debug("Getting protocol on link {} with Fmin {} and Rmin {}".format(nodes, Fmin, Rmin))
    if all([R < Rmin for _, R in link_properties]):
        logger.debug("Cannot satisfy rate {} between nodes {}".format(Rmin, nodes))
        return None

    if any([v < 1 for v in [nodeResources[n]['total'] for n in nodes]]):
        logger.debug("Not enough resources to generate link between nodes {}".format(nodes))
        return None

    # Check if any single link generation protocols exist
    for F, R in link_properties:
        if R >= Rmin and F >= Fmin:
            logger.debug("Link capable of generating without distillation using F={},R={}".format(F, R))
            return LinkProtocol(F=F, R=R, nodes=nodes)

    # logger.debug("Link not capable of generating without distillation")
    if any([v < 2 for v in [nodeResources[n]['total'] for n in nodes]]):
        logger.debug("Not enough resources to perform distillation with nodes {}".format(nodes))
        return None

    # Search for the link gen protocol with highest rate that satisfies fidelity
    bestR = Rmin
    bestProtocol = None

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
                currProtocol = DistillationProtocol(F=currF, R=currR, protocols=[currProtocol, linkProtocol], nodes=nodes)

            if (currProtocol.F > Fmin and currProtocol.R >= bestR) or (currProtocol.F >= Fmin and currProtocol.R > bestR):
                logger.debug("Found distillation protocol using F={},R={},numGens={}".format(currProtocol.F, currProtocol.R, numGens))
                bestR = currProtocol.R
                bestProtocol = currProtocol

    if bestProtocol is None:
        logger.debug("Failed to find distillation protocol")

    return bestProtocol
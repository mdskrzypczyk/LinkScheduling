from math import sqrt
from math import isclose
from copy import copy
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)


def esss(path, G, Fmin, Rmin):
    logger.debug("ESSS on path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
    if len(path) == 2:
        logger.debug("Down to elementary link, finding distillation protocol")
        link_properties = G.get_edge_data(*path)['capabilities']
        protocol = get_protocol_for_link(link_properties, Fmin, Rmin, path)
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
            possible_protocol, Rl, Rr = find_split_path_protocol(path, G, Fmin, Rmin, numL, numR)

            logger.info("{},{}".format(Rl, Rr))
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


def find_split_path_protocol(path, G, Fmin, Rmin, numL, numR):
    protocols = []
    maxDistillations = 10
    for num in range(maxDistillations):
        Rlink = Rmin * (num + 1)

        # Compute minimum fidelity in order for num distillations to achieve Fmin
        Fminswap = fidelity_for_distillations(num, Fmin)
        Funswapped = unswap_links(Fminswap)

        # Search for protocols on left and right that have above properties
        protocolL = esss(path[:numL], G, Funswapped, Rlink)
        protocolR = esss(path[-numR:], G, Funswapped, Rlink)

        # Add to list of protocols
        if protocolL is not None and protocolR is not None:
            logger.debug("Constructing protocol")
            Fswap = swap_links(protocolL.F, protocolR.F)
            Rswap = min(protocolL.R, protocolR.R)
            swap_protocol = SwapProtocol(F=Fswap, R=Rswap, protocols=[protocolL, protocolR], nodes=[path[-numR]])
            protocol = copy(swap_protocol)
            logger.debug("Swapped link F={}".format(Fswap))
            for i in range(num):
                Fdistilled = distill_links(protocol.F, Fswap)
                Rdistilled = Rswap / (i + 2)
                protocol = DistillationProtocol(F=Fdistilled, R=Rdistilled, protocols=[protocol, copy(swap_protocol)],
                                                nodes=[path[0], path[-1]])
                logger.debug("Distilled link F={}".format(Fdistilled))

            logger.debug("Found Swap/Distill protocol achieving F={},R={},numSwappedDistills={}".format(protocol.F, protocol.R,
                                                                                                 num + 1))
            logger.debug("Underlying link protocols have Fl={},Rl={} and Fr={},Rr={}".format(protocolL.F, protocolL.R,
                                                                                      protocolR.F, protocolR.R))
            protocols.append((protocol, protocolL.R, protocolR.R))

        else:
            Rl = 0 if not protocolL else protocolL.R
            Rr = 0 if not protocolR else protocolR.R
            protocols.append((Protocol(F=0, R=0, nodes=None), Rl, Rr))

    # Choose protocol with maximum rate > Rmin
    if protocols:
        protocol, Rl, Rr = sorted(protocols, key=lambda p: p[0].R)[-1]
        logger.debug("Found Swap/Distill protocol achieving F={},R={},numSwappedDistills={}".format(protocol.F, protocol.R,
                                                                                             num + 1))
        return protocol, Rl, Rr

    else:
        return None, 0, 0


def get_protocol_for_link(link_properties, Fmin, Rmin, nodes):
    if all([R < Rmin for _, R in link_properties]):
        return None

    # Check if any single link generation protocols exist
    for F, R in link_properties:
        if R >= Rmin and F >= Fmin:
            logger.debug("Link capable of generating without distillation using F={},R={}".format(F, R))
            return LinkProtocol(F=F, R=R, nodes=nodes)

    logger.debug("Link not capable of generating without distillation")
    # Search for the link gen protocol with highest rate that satisfies fidelity
    bestR = Rmin
    bestProtocol = None
    for F, R in link_properties:
        currF = F
        currR = R
        currProtocol = LinkProtocol(F=F, R=R, nodes=nodes)
        numGens = 1
        while currR >= Rmin and currF < Fmin:
            linkProtocol = LinkProtocol(F=F, R=R, nodes=nodes)
            currF = distill_links(currF, F)
            numGens += 1
            currR = R / numGens
            currProtocol = DistillationProtocol(F=currF, R=currR, protocols=[currProtocol, linkProtocol], nodes=nodes)

        if currProtocol.F >= Fmin and currProtocol.R >= bestR:
            logger.debug("Found distillation protocol using F={},R={},numGens={}".format(currProtocol.F, currProtocol.R, numGens))
            bestR = currProtocol.R
            bestProtocol = currProtocol

    return bestProtocol


def distill_links(F1, F2):
    a = F1 * F2
    b = (1 - F1) / 3
    c = (1 - F2) / 3
    return (a + b * c) / (a + F1 * c + F2 * b + 5 * b * c)


def undistill_link(Ftarget, Finitial=None):
    if Finitial is None:
        Finitial = Ftarget
    return (2 * Ftarget * Finitial - Finitial - 5 * Ftarget + 1) / (8 * Ftarget * Finitial - 10 * Finitial - 2 * Ftarget + 1)


def swap_links(F1, F2):
    return (4/3) * F1 * F2 - (1/3)*F1 - (1/3)*F2 + (1/3)


def unswap_links(Ftarget):
    a = 4 / 3
    b = -2 / 3
    c = (1/3) - Ftarget
    try:
        x1 = (-b + sqrt(b**2 - 4*a*c)) / (2*a)
        return x1
    except:
        import pdb
        pdb.set_trace()
        return 1


def distillations_for_fidelity(Finitial, Ftarget):
    if Finitial <= 0.5:
        return float('inf')
    else:
        num = 0
        currF = Finitial
        while Ftarget > currF:
            newF = distill_links(currF, Finitial)
            num += 1
            if newF <= currF:
                return float('inf')
            currF = newF
        return num


def fidelity_for_distillations(k, Ftarget):
    if k == 0:
        return Ftarget
    Fupper = Ftarget
    Flower = undistill_link(Ftarget)
    while not isclose(Flower, Fupper, abs_tol=0.00001):
        F = (Fupper + Flower) / 2
        j = distillations_for_fidelity(F, Ftarget)
        if j == k:
            return F
        elif j < k:
            Fupper = F
        else:
            Flower = F

    return 0


def max_distilled_fidelity(Finitial):
    if Finitial <= 0.5:
        return float('inf')
    else:
        currF = Finitial
        newF = distill_links(currF, Finitial)
        while newF > currF:
            currF = newF
            newF = distill_links(currF, Finitial)
        return currF


class Protocol:
    def __init__(self, F, R, nodes):
        self.F = F
        self.R = R
        self.nodes = nodes


class LinkProtocol(Protocol):
    name_template = "LG{};{};{}"
    count = 0

    def __init__(self, F, R, nodes):
        super(LinkProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.name = self.name_template.format(self.count, *nodes)
        LinkProtocol.count += 1

    def __copy__(self):
        return LinkProtocol(F=self.F, R=self.R, nodes=self.nodes)


class DistillationProtocol(LinkProtocol):
    name_template = "D{};{};{}"
    count = 0

    def __init__(self, F, R, protocols, nodes):
        super(DistillationProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.protocols = protocols
        self.name = self.name_template.format(self.count, *nodes)
        DistillationProtocol.count += 1

    def __copy__(self):
        return DistillationProtocol(F=self.F, R=self.R, nodes=self.nodes, protocols=[copy(p) for p in self.protocols])


class SwapProtocol(Protocol):
    name_template = "S{};{}"
    count = 0

    def __init__(self, F, R, protocols, nodes):
        super(SwapProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.protocols = protocols
        self.name = self.name_template.format(self.count, *nodes)
        SwapProtocol.count += 1

    def __copy__(self):
        return SwapProtocol(F=self.F, R=self.R, nodes=self.nodes, protocols=[copy(p) for p in self.protocols])

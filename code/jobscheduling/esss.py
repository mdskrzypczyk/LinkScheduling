from math import sqrt

def esss(path, G, Fmin, Rmin):
    if len(path) == 2:
        link_properties = G.get_edge(*path)
        protocol = get_protocol_for_link(link_properties, Fmin, Rmin)
        return protocol

    else:
        # Start by dividing the path in two
        numL = (len(path) + 1) // 2

        protocols = []
        maxDistillations = 3
        for num in range(maxDistillations):
            Rmin_link = Rmin / num

            # Compute minimum fidelity in order for num distillations to achieve Fmin
            ...

            # Search for protocols on left and right that have above properties
            ...

            # Add to list of protocols
            ...

        # Choose protocol with minimum rate > Rmin
        ...


def get_protocol_for_link(link_properties, Fmin, Rmin):
    # Check if any single link generation protocols exist
    for F, R in link_properties:
        if R >= Rmin and F >= Fmin:
            return LinkProtocol(F, R)

    # Search for the link gen protocol with highest rate that satisfies fidelity
    bestR = Rmin
    bestProtocol = None
    for F, R in link_properties:
        currF = F
        currR = R
        currProtocol = LinkProtocol(F=F, R=R)
        numGens = 1
        while currR >= Rmin and currF < Fmin:
            linkProtocol = LinkProtocol(F=F, R=R)
            currF = distill_link(currF, F)
            numGens += 1
            currR = R / numGens
            currProtocol = DistillationProtocol(F=currF, R=currR, protocols=[currProtocol, linkProtocol])

        if currProtocol.F >= Fmin and currProtocol.R >= bestR:
            bestR = currProtocol.R
            bestProtocol = currProtocol

    return bestProtocol


def distill_link(F1, F2):
    a = F1 * F2
    b = (1 - F1) / 3
    c = (1 - F2) / 3
    return (a + b * c) / (a + F1 * c + F2 * b + 5 * b * c)


def undistill_link(Ftarget, Finitial=None):
    if Finitial is None:
        Finitial = Ftarget
    return (2 * Ftarget * Finitial - Finitial - 5 * Ftarget + 1) / (8 * Ftarget*Finitial - 10*Finitial - 2*Ftarget + 1)


def distillations_for_fidelity(Finitial, Ftarget):
    if Finitial < 0.5:
        return float('inf')
    else:
        num = 0
        currF = Finitial
        while Ftarget > currF:
            newF = distill_link(currF, Finitial)
            num += 1
            if newF <= currF:
                print(num)
                return float('inf')
            print(newF)
            currF = newF
        return num


def fidelity_for_distillations(k, Ftarget):
    Finitial = undistill_link(Ftarget)
    for i in range(k):
        tmpFinitial = undistill_link(Ftarget, Finitial)
        Finitial = undistill_link(tmpFinitial)
    return Finitial


def swap_links(F1, F2):
    x = F1 * F2
    return x + (1 - F1 - F2 + x) / 3


class Protocol:
    def __init__(self, F, R):
        self.F = F
        self.R = R


class LinkProtocol(Protocol):
    def __init__(self, F, R):
        super(LinkProtocol, self).__init__(F, R)


class DistillationProtocol(Protocol):
    def __init__(self, F, R, protocols):
        super(DistillationProtocol, self).__init__(F, R)
        self.protocols = protocols


class SwapProtocol(Protocol):
    def __init__(self, F, R, protocols):
        super(SwapProtocol, self).__init__(F, R)
        self.protocols = protocols
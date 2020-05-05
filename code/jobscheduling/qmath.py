from math import sqrt, isclose


def distill_links(F1, F2):
    """
    Obtains the distilled fidelity of two links assuming Werner states
    :param F1: type float
        Fidelity of link 1
    :param F2: type float
        Fidelity of link 2
    :return: type float
        Fidelity of the distilled link
    """
    a = F1 * F2
    b = (1 - F1) / 3
    c = (1 - F2) / 3
    return (a + b * c) / (a + F1 * c + F2 * b + 5 * b * c)


def undistill_link(Ftarget, Finitial):
    """
    Obtains the fidelity of a link when reversing distillation of the target link assuming Werner states
    :param Ftarget: type float
        Fidelity of the distilled link
    :param Finitial: type float
        Fidelity of the second link used to produce the distilled link
    :return: type float
        Fidelity of the first link used to produce the distilled link
    """
    return (2 * Ftarget * Finitial - Finitial - 5 * Ftarget + 1) / \
           (8 * Ftarget * Finitial - 10 * Finitial - 2 * Ftarget + 1)


def undistill_link_even(Ftarget):
    """
    Reverses distilled fidelity calculation to obtain the fidelity of links used to distill. Assumes both links have
    the same fidelity assuming Werner states
    :param Ftarget: type float
        Fidelity of the distilled link
    :return: type float
        Fidelity of the links used to produce the distilled link
    """
    return (3 * sqrt(-4 * Ftarget**2 + 6 * Ftarget - 1) - 2 * Ftarget + 1) / (10 - 8 * Ftarget)


def swap_links(F1, F2):
    """
    Obtains the fidelity of a link produced using entanglement swapping assuming Werner states
    :param F1: type float
        Fidelity of link 1
    :param F2: type float
        Fidelity of link 2
    :return: type float
        Fidelity of the link produced by swapping link 1 and 2
    """
    return (4 / 3) * F1 * F2 - (1 / 3) * F1 - (1 / 3) * F2 + (1 / 3)


def unswap_links(Ftarget):
    """
    Obtains the fidelity of links used to produce the swapped link with fidelity Ftarget assuming Werner states
    :param Ftarget: type float
        The fidelity of the link produced using entanglement swapping
    :return: type float
        Fidelity of the links that produced a swapped link with fidelity Ftarget
    """
    a = 4 / 3
    b = -2 / 3
    c = (1 / 3) - Ftarget
    try:
        x1 = (-b + sqrt(b**2 - 4 * a * c)) / (2 * a)
        return x1
    except Exception:
        import pdb
        pdb.set_trace()
        return 1


def distillations_for_fidelity(Finitial, Ftarget):
    """
    Obtains the number of pumping distillations needed to reach a target fidelity using an initial link fidelity
    :param Finitial: type float
        The initial fidelity of the link used to pump
    :param Ftarget: type float
        The target fidelity desired by pumping using Finitial
    :return: type float
        The number of initial links needed to pump to Ftarget, float('inf') if not possible
    """
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
    """
    Obtains the required minimum fidelity needed to pump to a target fidelity only using k distillations
    :param k: type int
        The number of distillations allowed to be performed
    :param Ftarget: type float
        The desired target fidelity
    :return: type float
        The required initial link fidelity to obtain Ftarget by pumping k times
    """
    if k == 0:
        return Ftarget
    Fupper = Ftarget
    Flower = undistill_link(Ftarget, Ftarget)
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
    """
    Obtains the maximum distillable fidelity using entanglement pumping with an initial link fidelity
    :param Finitial: type float
        The initial link fidelity
    :return: type float
        The maximum achievable fidelity by pumping using the initial link fidelity
    """
    if Finitial <= 0.5:
        return float('inf')
    else:
        currF = Finitial
        newF = distill_links(currF, Finitial)
        while newF > currF:
            currF = newF
            newF = distill_links(currF, Finitial)
        return currF

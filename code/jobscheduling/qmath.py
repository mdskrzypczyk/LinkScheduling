from math import sqrt, isclose


def distill_links(F1, F2):
    a = F1 * F2
    b = (1 - F1) / 3
    c = (1 - F2) / 3
    return (a + b * c) / (a + F1 * c + F2 * b + 5 * b * c)


def undistill_link(Ftarget, Finitial):
    return (2 * Ftarget * Finitial - Finitial - 5 * Ftarget + 1) / (8 * Ftarget * Finitial - 10 * Finitial - 2 * Ftarget + 1)


def undistill_link_even(Ftarget):
    return (3*sqrt(-4*Ftarget**2 + 6*Ftarget -1) - 2*Ftarget + 1) / (10 - 8*Ftarget)


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


def unswap_three_links_equal_fidelity(Ftarget):
    return (1/4)*((3**(2/3))*(4*Ftarget-1)**(1/3) + 1)


def unswap_three_links(Fmiddle, Ftarget):
    return (3*sqrt((4*Fmiddle - 1)*(4*Ftarget - 1)) - 4*Fmiddle + 1) / (4 - 16*Fmiddle)


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
    if Finitial <= 0.5:
        return float('inf')
    else:
        currF = Finitial
        newF = distill_links(currF, Finitial)
        while newF > currF:
            currF = newF
            newF = distill_links(currF, Finitial)
        return currF
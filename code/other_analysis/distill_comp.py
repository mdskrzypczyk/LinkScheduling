import matplotlib.pyplot as plt
from jobscheduling.qmath import distill_links
from queue import PriorityQueue


def fidelity_vs_num_links():
    Flink = 0.9
    numlinks = 16
    xdata = range(1, numlinks + 1)

    # Plot pumping fidelity
    ydata_pump = []
    currF = Flink
    for _ in xdata:
        ydata_pump.append(currF)
        currF = distill_links(currF, Flink)

    # Plot nested pumping fidelity
    ydata_nested = []
    link_fidelities = PriorityQueue()
    for i in xdata:
        for _ in range(i):
            link_fidelities.put(Flink)
        while link_fidelities.qsize() > 1:
            F1 = link_fidelities.get()
            F2 = link_fidelities.get()
            link_fidelities.put(distill_links(F1, F2))
        currF = link_fidelities.get()
        ydata_nested.append(currF)

    plt.plot(xdata, ydata_pump, '-o', label="Standard")
    plt.plot(xdata, ydata_nested, '--o', label="Nested")
    plt.xlabel("Number of links", fontsize=16)
    plt.ylabel("Fidelity", fontsize=16)
    plt.title("Fidelity of Standard vs. Nested Pumping", fontsize=16)
    plt.legend(fontsize=16)
    plt.show()


if __name__ == "__main__":
    fidelity_vs_num_links()

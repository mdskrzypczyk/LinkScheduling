import itertools
import matplotlib
import matplotlib.pyplot as plt
from jobscheduling.log import LSLogger
from jobscheduling.protocols import schedule_dag_asap, convert_task_to_alap, shift_distillations_and_swaps
from jobscheduling.protocols import convert_protocol_to_task, schedule_dag_for_resources
from jobscheduling.visualize import schedule_and_resource_timelines, protocol_timeline, draw_DAG
from jobscheduling.topology import gen_line_topology
from simulations.common import get_protocol_without_rate_constraint


logger = LSLogger()

font = {'family': 'normal',
        'size': 14}

matplotlib.rc('font', **font)


def slot_size_selection():
    """
    Plots the latency and number of slots needed to encode repeater protocols for
    a few different levels of fidelity for both cases when two nodes are connected
    or separated by two hops
    :return: None
    """
    link_length = 5
    topology = gen_line_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, link_length=link_length)
    protocols = []

    source = '0'
    destinations = ['1', '2']
    fidelities = [0.6 + 0.1 * i for i in range(4)]
    for destination in destinations:
        for fidelity in fidelities:
            demand = (source, destination, fidelity, 1)
            protocol = get_protocol_without_rate_constraint(topology, demand)
            if protocol:
                print("Found protocol between {} and {} with fidelity {} and rate {}".format(source, destination,
                                                                                             protocol.F, protocol.R))
                protocols.append((demand, protocol))

    # Increments of 4ms
    slot_sizes = sorted(list(set([0.0001 * i for i in range(1, 200)])))
    latency_data = {}
    slot_count_data = {}
    for demand, protocol in protocols:
        pdata_lat = []
        pdata_slt = []
        for slot_size in slot_sizes:
            task = convert_protocol_to_task(demand, protocol, slot_size)
            task, dec, corr = schedule_dag_for_resources(task, topology)
            asap_d, alap_d, shift_d = dec
            if not corr:
                import pdb
                pdb.set_trace()
            elif asap_d < shift_d or alap_d < shift_d:
                import pdb
                pdb.set_trace()
            num_slots = task.c
            task_latency = num_slots * slot_size
            pdata_lat.append((slot_size, task_latency))
            pdata_slt.append((slot_size, num_slots))
            s, d, f, r = demand
            print("Hops: {}, Fidelity: {}, Slot Size: {}, Latency: {}".format(d, f, slot_size, task_latency))
        latency_data[demand] = pdata_lat
        slot_count_data[demand] = pdata_slt

    fig, axes = plt.subplots(nrows=1, ncols=4)
    for i, destination in enumerate(destinations):
        fmts = {
            0.6: ("-.", "0.8"),
            0.7: ("--", "0.6"),
            0.8: ("-", "0.4"),
            0.9: ("-", "0.2")
        }
        for demand in latency_data.keys():
            if demand[1] != destination:
                continue
            pdata = latency_data[demand]
            spdata = sorted(pdata)
            xdata = [d[0] for d in spdata]
            ydata = [d[1] for d in spdata]
            fidelity = round(demand[2], 2)
            label = "F={}".format(fidelity)
            fmt, c = fmts[fidelity]
            axes[i].plot(xdata, ydata, linestyle=fmt, color=c, label=label)
            axes[i].set(xlabel="Slot Size(s)", ylabel="Latency(s)")

        for demand in slot_count_data.keys():
            if demand[1] != destination:
                continue
            pdata = slot_count_data[demand]
            spdata = sorted(pdata)
            xdata = [d[0] for d in spdata]
            ydata = [d[1] for d in spdata]
            fidelity = round(demand[2], 2)
            label = "F={}".format(fidelity)
            fmt, c = fmts[fidelity]
            axes[i+2].plot(xdata, ydata, linestyle=fmt, color=c, label=label)
            axes[i+2].set(xlabel="Slot Size(s)", ylabel="Num Slots")

    axes[0].set_title("Link")
    axes[1].set_title("Two Hop")
    axes[2].set_title("Link")
    axes[3].set_title("Two Hop")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.96, 0.35), loc='lower right', fontsize=14)

    def on_resize(event):
        fig.tight_layout()
        fig.canvas.draw()

    fig.canvas.mpl_connect('resize_event', on_resize)
    plt.autoscale()
    plt.show()


def throughput_vs_chain_length():
    """
    Plots the achieved rate of a quantum repeater protocol for different levels of fidelity
    and path length between end nodes
    :return: None
    """
    num_network_nodes = 15
    link_length = 5
    topology = gen_line_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, link_length=link_length)
    for edge in topology[1].edges:
        topology[1].edges[edge]['capabilities'] = [(0.999, 1500)]
    protocols = []

    source = '0'
    destinations = [str(i) for i in range(3, num_network_nodes)]
    fidelities = [0.98, 0.985, 0.99, 0.995, 0.997]
    for destination in destinations:
        for fidelity in fidelities:
            print("Creating protocol betweein {} and {} with fidelity {}".format(source, destination, fidelity))
            demand = (source, destination, fidelity, 1e-10)
            protocol = get_protocol_without_rate_constraint(topology, demand)

            if protocol:
                protocols.append((demand, protocol))
            else:
                print("Failed to find protocol for demand {}".format(demand))
                protocols.append((demand, None))

    slot_size = 0.01
    latency_data = {}
    for demand, protocol in protocols:
        print("Processing demand {}".format(demand))
        if protocol is None:
            latency_data[demand] = 0
            continue

        task = convert_protocol_to_task(demand, protocol, slot_size)
        task, dec, corr = schedule_dag_for_resources(task, topology)
        asap_d, alap_d, shift_d = dec
        if not corr:
            import pdb
            pdb.set_trace()
        elif asap_d < shift_d or alap_d < shift_d:
            import pdb
            pdb.set_trace()
        task_latency = task.c * slot_size
        latency_data[demand] = 1 / task_latency
        s,d,f,r = demand
        print("Found protocol between {} and {} with fidelity {}, latency {}s, rate {}".format(s, d, f, task_latency, latency_data[demand]))

    for i, destination in enumerate(destinations):
        xdata = fidelities
        dest_demands = list(sorted(filter(lambda demand: demand[1] == destination, latency_data.keys())))
        ydata = [latency_data[demand] for demand in dest_demands]
        plt.plot(xdata, ydata, label="{} node chain".format(int(destination) + 1))

    plt.title("Repeater Protocol Rate vs. Fidelity")
    plt.xlabel("Fidelity")
    plt.ylabel("Rate (ebit/s)")
    plt.legend()
    plt.autoscale()
    plt.show()


def throughput_vs_link_length():
    """
    Plots the achieved rate of a quantum repeater protocol for different link lengths for a chain of 3 nodes
    :return: None
    """
    link_lengths = [5 + 5 * i for i in range(10)]
    fidelities = [0.55 + 0.05 * i for i in range(9)]
    latency_data = {}
    for length in link_lengths:
        topology = gen_line_topology(num_end_node_comm_q=1, num_end_node_storage_q=5, link_length=length)
        source = '0'
        destination = '2'
        protocols = []
        for fidelity in fidelities:
            demand = (source, destination, fidelity, 1)
            protocol = get_protocol_without_rate_constraint(topology, demand)
            if protocol:
                print("Found protocol between {} and {} with fidelity {} and rate {}".format(source, destination,
                                                                                             protocol.F, protocol.R))
                protocols.append((demand, protocol))
            else:
                protocols.append((demand, None))

        # Increments of 10ms
        slot_size = 0.01
        for demand, protocol in protocols:
            print("Processing demand {} with length {}".format(demand, length))
            if protocol is None:
                key = tuple(list(demand) + [length])
                latency_data[key] = 0
                continue

            task = convert_protocol_to_task(demand, protocol, slot_size)
            task, dec, corr = schedule_dag_for_resources(task, topology)
            asap_d, alap_d, shift_d = dec
            if not corr:
                import pdb
                pdb.set_trace()
            task_latency = task.c * slot_size
            key = tuple(list(demand) + [length])
            latency_data[key] = 1 / task_latency

    for length in link_lengths:
        xdata = fidelities
        length_demands = list(sorted(filter(lambda demand: demand[-1] == length, latency_data.keys())))
        ydata = [latency_data[demand] for demand in length_demands]
        plt.plot(xdata, ydata, label="{} km".format(length))

    plt.title("Repeater Protocol Rate vs. Fidelity")
    plt.xlabel("Fidelity")
    plt.ylabel("Rate (ebit/s)")
    plt.legend()
    plt.autoscale()
    plt.show()


def find_link_capabilities():
    """
    Function for checking what kinds of link capabilities end up in quantum repeater protocols generated
    :return: None
    """
    num_network_nodes = 6
    link_lengths = [5 + 5 * i for i in range(10)]
    fidelities = [0.55 + 0.05 * i for i in range(9)]
    link_capabilities = []

    for length in link_lengths:
        print("Processing link length: {}".format(length))
        topology = gen_line_topology(num_end_node_comm_q=1, num_end_node_storage_q=5, link_length=length)
        source = '0'
        for destination in [str(i) for i in range(1, num_network_nodes)]:
            print("Processing destination {}".format(destination))
            for fidelity in fidelities:
                demand = (source, destination, fidelity, 1)
                protocol = get_protocol_without_rate_constraint(topology, demand)
                if protocol:
                    q = [protocol]
                    while q:
                        next = q.pop(0)
                        if hasattr(next, "protocols"):
                            q += next.protocols
                        else:
                            cap = (length, round(next.F, 3), next.R)
                            if cap not in link_capabilities:
                                link_capabilities.append(cap)
                                print(type(next), cap)


def throughput_vs_resources():
    """
    Plots the achieved rate of a quantum repeater protocol for different combinations of # comm q/# storage q
    :return: None
    """
    link_length = 5
    num_comm_qubits = [1, 2, 4]
    num_storage_qubits = [3, 4, 5]
    fidelities = [0.55 + 0.05 * i for i in range(9)]
    latency_data = {}
    for num_comm, num_storage in list(itertools.product(num_comm_qubits, num_storage_qubits)):
        print("Using {} comm qs and {} storage qs".format(num_comm, num_storage))
        topology = gen_line_topology(num_end_node_comm_q=num_comm, num_end_node_storage_q=num_storage,
                                     link_length=link_length)
        source = '0'
        destination = '3'
        protocols = []
        for fidelity in fidelities:
            demand = (source, destination, fidelity, 1)
            protocol = get_protocol_without_rate_constraint(topology, demand)
            if protocol:
                print("Found protocol between {} and {} with fidelity {} and rate {}".format(source, destination,
                                                                                             protocol.F, protocol.R))
                protocols.append((demand, protocol))
            else:
                protocols.append((demand, None))

        # Increments of 10ms
        slot_size = 0.01
        for demand, protocol in protocols:
            print("Processing demand {}".format(demand))
            if protocol is None:
                key = tuple(list(demand) + [num_comm, num_storage])
                latency_data[key] = 0
                continue

            task = convert_protocol_to_task(demand, protocol, slot_size)
            task, dec, corr = schedule_dag_for_resources(task, topology)
            asap_d, alap_d, shift_d = dec
            if not corr:
                import pdb
                pdb.set_trace()
            elif asap_d < shift_d or alap_d < shift_d:
                import pdb
                pdb.set_trace()
            task_latency = task.c * slot_size
            key = tuple(list(demand) + [num_comm, num_storage])
            latency_data[key] = 1 / task_latency

    for num_comm, num_storage in list(itertools.product(num_comm_qubits, num_storage_qubits)):
        xdata = fidelities
        length_demands = list(sorted(filter(lambda demand: demand[-2] == num_comm and demand[-1] == num_storage,
                                            latency_data.keys())))
        ydata = [latency_data[demand] for demand in length_demands]
        plt.plot(xdata, ydata, label="{} Comm. {} Stor.".format(num_comm, num_storage))

    plt.title("Repeater Protocol Rate vs. Fidelity")
    plt.xlabel("Fidelity")
    plt.ylabel("Rate (ebit/s)")
    plt.legend()
    plt.autoscale()
    plt.show()


def visualize_protocol_scheduling():
    """
    Function for visualizing the temporal scheduling of a repeater protocol under different resources
    :return: None
    """
    line_topology = gen_line_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, link_length=5)
    demand = ('0', '3', 0.55, 0.01)
    protocol = get_protocol_without_rate_constraint(line_topology, demand)
    task = convert_protocol_to_task(demand, protocol, 0.01)
    asap_latency, asap_decoherence, asap_correct = schedule_dag_asap(task, line_topology)
    alap_latency, alap_decoherence, alap_correct = convert_task_to_alap(task)
    shift_latency, shift_decoherence, shift_correct = shift_distillations_and_swaps(task)
    # draw_DAG(task)
    # import pdb
    # pdb.set_trace()
    protocol_timeline(task)
    import pdb
    pdb.set_trace()
    demand = ('0', '2', 0.8, 0.01)
    line_topology = gen_line_topology(num_end_node_comm_q=2, num_end_node_storage_q=3, link_length=5)
    protocol = get_protocol_without_rate_constraint(line_topology, demand)
    task = convert_protocol_to_task(demand, protocol, 0.01)
    asap_latency, asap_decoherence, asap_correct = schedule_dag_asap(task, line_topology)
    alap_latency, alap_decoherence, alap_correct = convert_task_to_alap(task)
    shift_latency, shift_decoherence, shift_correct = shift_distillations_and_swaps(task)
    protocol_timeline(task)
    line_topology = gen_line_topology(num_end_node_comm_q=4, num_end_node_storage_q=3, link_length=5)
    protocol = get_protocol_without_rate_constraint(line_topology, demand)
    task = convert_protocol_to_task(demand, protocol, 0.01)
    asap_latency, asap_decoherence, asap_correct = schedule_dag_asap(task, line_topology)
    alap_latency, alap_decoherence, alap_correct = convert_task_to_alap(task)
    shift_latency, shift_decoherence, shift_correct = shift_distillations_and_swaps(task)
    protocol_timeline(task)


def visualize_scheduled_protocols():
    """
    Function for visualizing the temporal scheduling of a repeater protocol under different path lengths, resource
    counts, and link lengths
    :return: None
    """
    # Iterate over path length
    max_num_nodes = 6
    max_num_resources = 8
    link_lengths = range(5, 10, 5)
    fidelities = [0.75, 0.8, 0.85, 0.9]
    slot_size = 0.05
    data = []
    for num_nodes in range(3, max_num_nodes):
        for num_resources in range(1, max_num_resources):
            # Iterate over the different lengths for links (make line equidistant
            for length in link_lengths:
                # Construct topology
                line_topology = gen_line_topology(num_end_node_comm=num_resources, num_end_node_storage=num_resources,
                                                  link_length=length)
                # Iterate over the different fidelities
                for Fmin in fidelities:
                    print("Collecting ({}, {}, {}, {})".format(num_nodes, num_resources, length, Fmin))
                    demand = (str(0), str(num_nodes - 1), Fmin, 0.01)
                    protocol = get_protocol_without_rate_constraint(line_topology, demand)
                    if protocol is None:
                        data.append((length, num_nodes, num_resources, Fmin, None))
                        logger.warning("Demand {} could not be satisfied!".format(demand))
                    else:
                        task = convert_protocol_to_task(demand, protocol, slot_size)
                        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, line_topology)
                        data.append((length, num_nodes, num_resources, Fmin, scheduled_task))

    for entry in sorted(data):
        length, num_nodes, num_resources, Fmin, task = entry
        if task:
            print(entry)
            schedule = [(0, task.c, task)]
            schedule_and_resource_timelines([task], schedule)


if __name__ == "__main__":
    # slot_size_selection()
    throughput_vs_chain_length()
    # throughput_vs_link_length()
    # throughput_vs_resources()
    # visualize_protocol_scheduling()
    # find_link_capabilities()

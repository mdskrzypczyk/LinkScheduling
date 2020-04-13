import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from jobscheduling.log import LSLogger
from jobscheduling.protocols import schedule_dag_asap, convert_task_to_alap, shift_distillations_and_swaps
from jobscheduling.protocols import convert_protocol_to_task, schedule_dag_for_resources
from jobscheduling.schedulers.NPEDF import MultipleResourceNonBlockNPEDFScheduler
from jobscheduling.visualize import draw_DAG, schedule_timeline, resource_timeline, schedule_and_resource_timelines, protocol_timeline
from jobscheduling.topology import gen_line_topology
from simulations.common import get_protocol_without_rate_constraint


logger = LSLogger()


def slot_size_selection():
    num_network_nodes = 5
    link_length = 5
    topology = gen_line_topology(num_network_nodes, num_comm_q=1, num_storage_q=5, link_distance=link_length)
    protocols = []

    source = '0'
    destinations = ['1', '2']
    fidelities = [0.6 + 0.1*i for i in range(4)]
    for destination in destinations:
        for fidelity in fidelities:
            demand = (source, destination, fidelity, 1)
            protocol = get_protocol_without_rate_constraint(topology, demand)
            if protocol:
                print("Found protocol between {} and {} with fidelity {} and rate {}".format(source, destination,
                                                                                             protocol.F, protocol.R))
                protocols.append((demand, protocol))

    # Increments of 4ms
    slot_sizes = sorted(list(set([0.004*i for i in range(1, 50)])))
    latency_data = {}
    slot_count_data = {}
    for demand, protocol in protocols:
        pdata_lat = []
        pdata_slt = []
        print("Processing demand {}".format(demand))
        for slot_size in slot_sizes:
            print("Processing slot size {}".format(slot_size))
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
        latency_data[demand] = pdata_lat
        slot_count_data[demand] = pdata_slt

    figure, axes = plt.subplots(nrows=2, ncols=2)
    for i, destination in enumerate(destinations):
        for demand, pdata in latency_data.items():
            if demand[1] != destination:
                continue
            spdata = sorted(pdata)
            xdata = [d[0] for d in spdata]
            ydata = [d[1] for d in spdata]
            label = "F={}".format(round(demand[2], 2))
            axes[0, i].plot(xdata, ydata, label=label)
        axes[0, i].set(xlabel="Slot Size(s)", ylabel="Latency (s)")

        for demand, pdata in slot_count_data.items():
            if demand[1] != destination:
                continue
            spdata = sorted(pdata)
            xdata = [d[0] for d in spdata]
            ydata = [d[1] for d in spdata]
            label = "F={}".format(round(demand[2], 2))
            axes[1, i].plot(xdata, ydata, label=label)

        axes[1, i].set(xlabel="Slot Size(s)", ylabel="Num Slots")

    axes[0, 0].set_title("Link")
    axes[0, 1].set_title("One Hop")
    plt.legend()
    plt.autoscale()
    plt.show()


def throughput_vs_chain_length():
    num_network_nodes = 6
    link_length = 5
    topology = gen_line_topology(num_network_nodes, num_comm_q=1, num_storage_q=5, link_distance=link_length)
    protocols = []

    source = '0'
    destinations = [str(i) for i in range(2, num_network_nodes)]
    fidelities = [0.55 + 0.05 * i for i in range(9)]
    for destination in destinations:
        for fidelity in fidelities:
            demand = (source, destination, fidelity, 1)
            protocol = get_protocol_without_rate_constraint(topology, demand)

            if protocol:
                print("Found protocol between {} and {} with fidelity {} and rate {}".format(source, destination,
                                                                                             protocol.F, protocol.R))
                protocols.append((demand, protocol))
            else:
                protocols.append((demand, None))

    # Increments of 4ms
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

    for i, destination in enumerate(destinations):
        xdata = fidelities
        dest_demands = list(sorted(filter(lambda demand: demand[1] == destination, latency_data.keys())))
        ydata = [latency_data[demand] for demand in dest_demands]
        plt.plot(xdata, ydata, label="{} hops".format(i+1))

    plt.title("Repeater Protocol Rate vs. Fidelity")
    plt.xlabel("Fidelity")
    plt.ylabel("Rate (ebit/s)")
    plt.legend()
    plt.autoscale()
    plt.show()


def throughput_vs_link_length():
    num_network_nodes = 3
    link_lengths = [5 + 5*i for i in range(10)]
    fidelities = [0.5 + 0.05*i for i in range(10)]
    protocols = []
    latency_data = {}
    for length in link_lengths:
        topology = gen_line_topology(num_network_nodes, num_comm_q=1, num_storage_q=5, link_distance=length)
        source = '0'
        destination = '2'
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
        plt.plot(xdata, ydata, label="{}km".format(length))

    plt.title("Repeater Protocol Rate vs. Fidelity")
    plt.xlabel("Fidelity")
    plt.ylabel("Rate (ebit/s)")
    plt.legend()
    plt.autoscale()
    plt.show()


def throughput_vs_resources():
    num_network_nodes = 4
    link_length = 5
    num_comm_qubits = [1, 2]
    num_storage_qubits = [2, 4]
    fidelities = [0.5 + 0.05*i for i in range(10)]
    latency_data = {}
    for num_comm, num_storage in list(itertools.product(num_comm_qubits, num_storage_qubits)):
        print("Using {} comm qs and {} storage qs".format(num_comm, num_storage))
        topology = gen_line_topology(num_network_nodes, num_comm_q=num_comm, num_storage_q=num_storage, link_distance=link_length)
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
        length_demands = list(sorted(filter(lambda demand: demand[-2] == num_comm and demand[-1] == num_storage, latency_data.keys())))
        ydata = [latency_data[demand] for demand in length_demands]
        plt.plot(xdata, ydata, label="{} Comm. {} Stor.".format(num_comm, num_storage))

    plt.title("Repeater Protocol Rate vs. Fidelity")
    plt.xlabel("Fidelity")
    plt.ylabel("Rate (ebit/s)")
    plt.legend()
    plt.autoscale()
    plt.show()


def example_schedule():
    network_topologies = gen_topologies(9, num_comm_q=1, num_storage_q=1, link_distance=5)
    grid_topology = network_topologies[2]
    demands = [('0,1', '2,1', 0.8, 1), ('1,0', '1,2', 0.8, 1)]
    taskset = []
    for demand in demands:
        protocol = get_protocol_without_rate_constraint(grid_topology, demand)
        task = convert_protocol_to_task(demand, protocol, 0.05)
        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, grid_topology)
        taskset.append(scheduled_task)

    scheduler = MultipleResourceNonBlockNPEDFScheduler()
    schedule = scheduler.schedule_tasks(taskset, grid_topology)
    import pdb
    pdb.set_trace()
    sub_taskset, sub_schedule, _ = schedule[0]
    schedule_and_resource_timelines(sub_taskset, sub_schedule)


def visualize_protocol_scheduling():
    line_topology = gen_line_topology(3, num_comm_q=2, num_storage_q=4, link_distance=5)
    demand = ('0', '2', 0.8, 0.01)
    protocol = get_protocol_without_rate_constraint(line_topology, demand)
    task = convert_protocol_to_task(demand, protocol, 0.01)
    asap_latency, asap_decoherence, asap_correct = schedule_dag_asap(task, line_topology)
    alap_latency, alap_decoherence, alap_correct = convert_task_to_alap(task)
    shift_latency, shift_decoherence, shift_correct = shift_distillations_and_swaps(task)
    protocol_timeline(task)

    line_topology = gen_line_topology(5, num_comm_q=10, num_storage_q=4, link_distance=5)
    import pdb
    pdb.set_trace()
    protocol = get_protocol_without_rate_constraint(line_topology, demand)
    task = convert_protocol_to_task(demand, protocol, 0.01)
    asap_latency, asap_decoherence, asap_correct = schedule_dag_asap(task, line_topology)
    protocol_timeline(task)
    alap_latency, alap_decoherence, alap_correct = convert_task_to_alap(task)
    protocol_timeline(task)
    shift_latency, shift_decoherence, shift_correct = shift_distillations_and_swaps(task)
    protocol_timeline(task)


def visualize_scheduled_protocols():
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
                network_topologies = gen_topologies(num_nodes, num_comm_q=num_resources,
                                                    num_storage_q=num_resources, link_distance=length)

                line_topology = network_topologies[0]
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
                        latency = scheduled_task.c * slot_size
                        achieved_rate = 1 / latency
                        data.append((length, num_nodes, num_resources, Fmin, scheduled_task))


    for entry in sorted(data):
        length, num_nodes, num_resources, Fmin, task = entry
        if task:
            print(entry)
            schedule = [(0, task.c, task)]
            schedule_and_resource_timelines([task], schedule)


def plot_results(data):
    schedulers = list(data["(2, 4)"].keys())
    for res_conf, res_conf_data in data.items():
        for metric in ["throughput", "wcrt", "jitter", "num_demands"]:
            means = defaultdict(list)
            for sched in schedulers:
                sched_data = res_conf_data[sched]
                for fidelity, fidelity_data in sched_data.items():
                    means[fidelity].append(fidelity_data[metric])

            labels = [''.join([c for c in sched if c.isupper()]) for sched in schedulers]
            x = np.arange(len(labels))  # the label locations
            total_width = 0.7       # Width of all bars
            width = total_width / len(means.keys())   # the width of the bars

            fig, ax = plt.subplots()
            offset = (len(means.keys()) - 1) * width / 2
            for i, fidelity in enumerate(means.keys()):
                ax.bar(x - offset + i*width, means[fidelity], width, label="F={}".format(fidelity))

            # Add some text for labels, title and custom x-axis tick labels, etc.
            metric_to_label = {"throughput": "Throughput", "wcrt": "Worst-case Response Time", "jitter": "Jitter",
                               "num_demands": "Satisfied Demands"}
            metric_to_units = {"throughput": "(ebit/s)", "wcrt": "(s)", "jitter": "(s^2)", "num_demands": ""}
            ax.set_ylabel("{} {}".format(metric_to_label[metric], metric_to_units[metric]))
            ax.set_title('{} by scheduler and fidelity {}'.format(metric_to_label[metric], res_conf))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            fig.tight_layout()

            plt.show()


if __name__ == "__main__":
    # slot_size_selection()
    # throughput_vs_chain_length()
    # throughput_vs_link_length()
    # throughput_vs_resources()
    visualize_protocol_scheduling()

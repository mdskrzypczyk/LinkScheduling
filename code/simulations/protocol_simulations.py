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
    topology = gen_line_topology(num_network_nodes, num_comm_q=1, num_storage_q=3, link_distance=link_length)
    protocols = []

    source = '0'
    destination = '4'
    fidelities = [0.55 + 0.05*i for i in range(7)]
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
            num_slots = (task.sinks[0].a + task.sinks[0].c)
            task_latency = num_slots * slot_size
            pdata_lat.append((slot_size, task_latency))
            pdata_slt.append((slot_size, num_slots))
        latency_data[demand] = pdata_lat
        slot_count_data[demand] = pdata_slt

    for demand, pdata in latency_data.items():
        spdata = sorted(pdata)
        xdata = [d[0] for d in spdata]
        ydata = [d[1] for d in spdata]
        label = "F={}".format(round(demand[2], 2))
        plt.plot(xdata, ydata, label=label)

    plt.legend()
    plt.autoscale()
    plt.xlabel("Slot Size (s)")
    plt.ylabel("Latency (s)")
    plt.title("Protocol Latency vs. Slot Size")
    plt.show()

    import pdb
    pdb.set_trace()

    for demand, pdata in slot_count_data.items():
        pdata = list(sorted(filter(lambda d: d[0] <= 0.01, pdata)))
        spdata = sorted(pdata)
        xdata = [d[0] for d in spdata]
        ydata = [d[1] for d in spdata]
        label = "F={}".format(round(demand[2], 2))
        plt.plot(xdata, ydata, label=label)

    plt.legend()
    plt.autoscale()
    plt.xlabel("Slot Size (s)")
    plt.ylabel("Latency (# slots)")
    plt.title("Protocol Latency vs. Slot Size")
    plt.show()


def throughput_vs_link_length():
    # Iterate over path length
    max_num_nodes = 6
    max_num_resources = 8
    link_lengths = range(5, 25, 5)
    fidelities = [0.75, 0.8, 0.85, 0.9]
    slot_size = 0.05
    data = []
    for num_nodes in range(2, max_num_nodes):
        for num_resources in range(1, max_num_resources):
            # Iterate over the different lengths for links (make line equidistant
            for length in link_lengths:
                # Construct topology
                network_topologies = gen_topologies(num_nodes, num_comm_q=num_resources,
                                                    num_storage_q=num_resources, link_distance=length)

                line_topology = network_topologies[0]
                #Iterate over the different fidelities
                for Fmin in fidelities:
                    print("Collecting ({}, {}, {}, {})".format(num_nodes, num_resources, length, Fmin))
                    demand = (str(0), str(num_nodes-1), Fmin, 0.01)
                    protocol = get_protocol_without_rate_constraint(line_topology, demand)
                    if protocol is None:
                        data.append((length, num_nodes, num_resources, Fmin, 0))
                        logger.warning("Demand {} could not be satisfied!".format(demand))
                    else:
                        task = convert_protocol_to_task(demand, protocol, slot_size)
                        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, line_topology)
                        latency = scheduled_task.c * slot_size
                        achieved_rate = 1 / latency
                        data.append((length, num_nodes, num_resources, Fmin, achieved_rate))

    with open("throughput_v_path_length.dat", "w") as f:
        for datapoint in data:
            f.write("{}\n".format(datapoint))

    # Fix the fidelities and link length, plot chain length vs achieved rate
    for length in link_lengths:
        for fidelity in fidelities:
            for num_resources in range(1, max_num_resources):
                matching_data = list(sorted(filter(lambda entry: entry[0] == length and entry[2] == num_resources and entry[3] == fidelity, data), key=lambda entry: entry[1]))
                xdata = [entry[1] for entry in matching_data]
                ydata = [entry[4] for entry in matching_data]
                plt.plot(xdata, ydata, label="F={},L={},C={}".format(fidelity, length, num_resources))
            plt.xlabel("Number of hops")
            plt.ylabel("Rate")
            plt.legend()
            plt.show()


def throughput_vs_chain_length():
    pass

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
    network_topologies = gen_topologies(10, num_comm_q=1, num_storage_q=3, link_distance=5)
    line_topology = network_topologies[0]
    # demand = ('0', '2', 0.8, 1)
    demand = ('4', '2', 0.879, 0.01953125)
    protocol = get_protocol_without_rate_constraint(line_topology, demand)
    task = convert_protocol_to_task(demand, protocol, 0.1)
    draw_DAG(task, view=True)
    asap_latency, asap_decoherence, asap_correct = schedule_dag_asap(task, line_topology)
    protocol_timeline(task)
    alap_latency, alap_decoherence, alap_correct = convert_task_to_alap(task)
    protocol_timeline(task)
    shift_latency, shift_decoherence, shift_correct = shift_distillations_and_swaps(task)
    protocol_timeline(task)
    import pdb
    pdb.set_trace()


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
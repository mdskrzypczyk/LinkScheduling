import matplotlib.pyplot as plt
import networkx as nx
import random
import time
from math import sqrt
from collections import defaultdict
from intervaltree import IntervalTree, Interval
from device_characteristics.nv_links import load_link_data
from jobscheduling.log import LSLogger
from jobscheduling.protocolgen import create_protocol, LinkProtocol, DistillationProtocol, SwapProtocol
from jobscheduling.protocols import convert_protocol_to_task, schedule_dag_for_resources
from jobscheduling.task import get_lcm_for, get_gcd_for
from jobscheduling.schedulers.ClockDriven import UniResourceFlowScheduler
from jobscheduling.schedulers.BlockNPEDF import UniResourceBlockNPEDFScheduler, MultipleResourceBlockNPEDFScheduler
from jobscheduling.schedulers.BlockNPRM import UniResourceBlockNPRMScheduler, MultipleResourceBlockNPRMScheduler
from jobscheduling.schedulers.BlockPBEDF import UniResourcePreemptionBudgetScheduler,\
    UniResourceFixedPointPreemptionBudgetScheduler, UniResourceConsiderateFixedPointPreemptionBudgetScheduler
from jobscheduling.schedulers.SearchBlockPBEDF import MultipleResourceInconsiderateBlockPreemptionBudgetScheduler,\
    MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler, MultipleResourceConsiderateBlockPreemptionBudgetScheduler,\
    MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler, MultipleResourceInconsiderateSegmentPreemptionBudgetScheduler,\
    MultipleResourceConsiderateSegmentPreemptionBudgetScheduler
from jobscheduling.schedulers.CEDF import UniResourceCEDFScheduler, MultipleResourceBlockCEDFScheduler
from jobscheduling.schedulers.NPEDF import MultipleResourceNonBlockNPEDFScheduler
from jobscheduling.schedulers.NPRM import MultipleResourceNonBlockNPRMScheduler
from jobscheduling.schedulers.ILP import MultipleResourceILPBlockNPEDFScheduler
from jobscheduling.visualize import draw_DAG, schedule_timeline, resource_timeline, schedule_and_resource_timelines, protocol_timeline

logger = LSLogger()


def get_dimensions(n):
    divisors = []
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
            divisors.append(currentDiv+1)

    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i]-sqrt(n)))
    wIndex = hIndex + 1

    return divisors[hIndex], divisors[wIndex]


def gen_topologies(n, num_comm_q=2, num_storage_q=2, link_distance=5):
    d_to_cap = load_link_data()
    link_capabilities = [(d, d_to_cap[str(d)]) for d in [5]]
    link_capability = d_to_cap[str(link_distance)]
    # Line
    lineGcq = nx.Graph()
    lineG = nx.Graph()
    for i in range(n):
        comm_qs = []
        storage_qs = []
        for c in range(num_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        lineGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        lineG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)
        if i > 0:
            prev_node_id = i - 1
            for j in range(num_comm_q):
                for k in range(num_comm_q):
                    lineGcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(i, k))

            lineG.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability, weight=link_distance)

    # Ring
    ringGcq = nx.Graph()
    ringG = nx.Graph()
    for i in range(n):
        comm_qs = []
        storage_qs = []
        for c in range(num_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        ringGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        ringG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)
        if i > 0:
            prev_node_id = i - 1
            for j in range(num_comm_q):
                for k in range(num_comm_q):
                    ringGcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(i, k))
            ringG.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability, weight=link_distance)

    ringG.add_edge("{}".format(0), "{}".format(n-1), capabilities=link_capability, weight=link_distance)

    # Demo
    demoGcq = nx.Graph()
    demoG = nx.Graph()
    for i in range(4):
        comm_qs = []
        storage_qs = []
        for c in range(num_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        demoGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        demoG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)
        if i > 0:
            prev_node_id = i - 1
            for j in range(num_comm_q):
                for k in range(num_comm_q):
                    demoGcq.add_edge("{}-C{}".format(prev_node_id, j), "{}-C{}".format(i, k))

    demoG.add_edge("0", "1", capabilities=d_to_cap["10"], weight=10)
    demoG.add_edge("1", "2", capabilities=d_to_cap["15"], weight=15)
    demoG.add_edge("2", "3", capabilities=d_to_cap["35"], weight=35)
    demoG.add_edge("3", "0", capabilities=d_to_cap["50"], weight=50)

    for j in range(1):
        for k in range(1):
            ringGcq.add_edge("{}-{}".format(0, j), "{}-{}".format(n-1, k), capabilities=link_capability, weight=link_distance)

    # Grid
    w, h = get_dimensions(n)
    gridGcq = nx.Graph()
    gridG = nx.Graph()
    for i in range(w):
        for j in range(h):
            comm_qs = []
            storage_qs = []
            for c in range(num_comm_q):
                comm_q_id = "{},{}-C{}".format(i, j, c)
                comm_qs.append(comm_q_id)
            for s in range(num_storage_q):
                storage_q_id = "{},{}-S{}".format(i, j, s)
                storage_qs.append(storage_q_id)
            gridGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
            gridG.add_node("{},{}".format(i, j), comm_qs=comm_qs, storage_qs=storage_qs)

            # Connect upward
            if j > 0:
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i, j-1), capabilities=link_capability,
                               weight=link_distance)
                for k in range(num_comm_q):
                    for l in range(num_comm_q):
                        gridGcq.add_edge("{},{}-C{}".format(i, j-1, k), "{},{}-C{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance)
            # Connect left
            if i > 0:
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i - 1, j), capabilities=link_capability,
                               weight=link_distance)
                for k in range(num_comm_q):
                    for l in range(num_comm_q):
                        gridGcq.add_edge("{},{}-C{}".format(i-1, j, k), "{},{}-C{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance)

    return [(lineGcq, lineG), (ringGcq, ringG), (gridGcq, gridG), (demoGcq, demoG)]


def gen_plus_topology(num_nodes=5, end_node_resources=(1, 3), center_resources=(1, 3), link_distance=5):
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_distance)]
    # Line
    starGcq = nx.Graph()
    starG = nx.Graph()

    # First make the center
    num_comm_center, num_storage_center = center_resources
    comm_qs = []
    storage_qs = []
    i = num_nodes - 1
    for c in range(num_comm_center):
        comm_q_id = "{}-C{}".format(i, c)
        comm_qs.append(comm_q_id)
    for s in range(num_storage_center):
        storage_q_id = "{}-S{}".format(i, s)
        storage_qs.append(storage_q_id)
    starGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
    starG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)

    # Then make the end nodes
    num_comm_end_node, num_storage_end_node = end_node_resources
    for i in range(num_nodes - 1):
        comm_qs = []
        storage_qs = []
        for c in range(num_comm_end_node):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_storage_end_node):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        starGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        starG.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs)

        center_node_id = num_nodes - 1
        for j in range(num_comm_center):
            for k in range(num_comm_end_node):
                starGcq.add_edge("{}-C{}".format(center_node_id, j), "{}-C{}".format(i, k))

        starG.add_edge("{}".format(center_node_id), "{}".format(i), capabilities=link_capability,
                       weight=link_distance)

    return starGcq, starG


def get_schedulers():
    schedulers = [
        # UniResourceFlowScheduler,
        # UniResourcePreemptionBudgetScheduler,
        # UniResourceFixedPointPreemptionBudgetScheduler,
        # UniResourceConsiderateFixedPointPreemptionBudgetScheduler,
        # UniResourceBlockNPEDFScheduler,
        # UniResourceBlockNPRMScheduler,
        # UniResourceCEDFScheduler,
        # MultipleResourceILPBlockNPEDFScheduler,
        MultipleResourceBlockCEDFScheduler,
        MultipleResourceBlockNPEDFScheduler,
        MultipleResourceBlockNPRMScheduler,
        # MultipleResourceInconsiderateBlockPreemptionBudgetScheduler,
        # MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler,
        # MultipleResourceInconsiderateSegmentPreemptionBudgetScheduler,
        # MultipleResourceConsiderateBlockPreemptionBudgetScheduler,
        # MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler,
        # MultipleResourceConsiderateSegmentPreemptionBudgetScheduler,
        MultipleResourceNonBlockNPEDFScheduler,
        MultipleResourceNonBlockNPRMScheduler,
    ]
    return schedulers


def get_network_demands(network_topology, num):
    _, nodeG = network_topology
    demands = []
    for num_demands in range(num):
        src, dst = random.sample(nodeG.nodes, 2)
        fidelity = round(0.6 + random.random() * (3 / 10), 3)                    # Fidelity range between F=0.6 and 1
        rate = 10 / (2**random.choice([i for i in range(7, 11)]))       # Rate range between 0.2 and 1
        demands.append((src, dst, fidelity, rate))
    return demands


def get_protocol(network_topology, demand):
    s, d, f, r = demand
    _, nodeG = network_topology
    path = nx.shortest_path(G=nodeG, source=s, target=d, weight="weight")
    protocol = create_protocol(path, nodeG, f, r)

    if protocol:
        logger.debug("Found protocol that satisfies demands")
        return protocol

    else:
        return None

    logger.debug("Trying to find protocol without rate constraint")
    protocol = create_protocol(path, nodeG, f, 1e-10)
    if protocol:
        logger.warning("Found protocol without rate constraint")
        return protocol

    logger.debug("Trying to find protocol without fidelity/rate constraints")
    protocol = create_protocol(path, nodeG, 0.5, 0)
    if protocol:
        logger.warning("Found protocol without fidelity/rate constraint")
        return protocol

    return None


def get_protocol_without_rate_constraint(network_topology, demand):
    s, d, f, r = demand
    _, nodeG = network_topology
    path = nx.shortest_path(G=nodeG, source=s, target=d, weight="weight")
    protocol = create_protocol(path, nodeG, f, 1e-10)
    if protocol:
        logger.warning("Found protocol without rate constraint")
        return protocol
    else:
        return None


def select_rate(achieved_rate):
    rates = [10 / (2 ** i) for i in range(7, 11)]  # Rate range between 0.2 and 1
    rates = list(filter(lambda r: r < achieved_rate, rates))
    return random.choice(rates)


def get_network_resources(topology):
    _, nodeG = topology
    network_resources = {}
    network_nodes = nodeG.nodes
    for node in network_nodes:
        numCommResources = len(nodeG.nodes[node]['comm_qs'])
        numStorResources = len(nodeG.nodes[node]['storage_qs'])
        network_resources[node] = {
            "comm": numCommResources,
            "storage": numStorResources,
            "total": numCommResources + numStorResources
        }
    return network_resources


def check_resource_utilization(taskset, network_resources):
    resource_utilization = defaultdict(float)
    for task in taskset:
        resource_intervals = task.get_resource_intervals()
        for resource, itree in resource_intervals.items():
            utilization = sum([i.end - i.begin for i in itree]) / task.p
            resource_string = get_resource_string(resource)
            resource_utilization[resource_string] += utilization
            node, rtype = resource_string
            max_utilization = network_resources[node]["comm"] if rtype == "C" else network_resources[node]["storage"]
            if resource_utilization[resource_string] > max_utilization:
                logger.warning("Taskset overutilizes resource {}, computed {} where limit is {}".format(resource_string,
                                                                                                        resource_utilization[resource_string],
                                                                                                        max_utilization))
                return False

    for resource in resource_utilization.keys():
        resource_utilization[resource] = round(resource_utilization[resource], 3)

    logger.info("Resource utilization: {}".format(resource_utilization))
    return True


def get_resource_string(resource):
    resource_node, resource_id = resource.split('-')
    resource_type = resource_id[0]
    return resource_node + resource_type


def verify_schedule(tasks, schedule):
    global_resource_intervals = defaultdict(IntervalTree)
    for start, end, t in schedule:
        task_resource_intervals = t.get_resource_intervals()
        for resource, itree in task_resource_intervals.items():
            offset_itree = IntervalTree([Interval(i.begin + start, i.end + start) for i in itree])
            for interval in offset_itree:
                if global_resource_intervals[resource].overlap(interval.begin, interval.end):
                    import pdb
                    pdb.set_trace()
                    return False
                global_resource_intervals[resource].add(interval)

    return True


def slot_size_selection():
    num_network_nodes = 4
    link_distances = [5] #list(range(5, 55, 5))
    for i in range(1, num_network_nodes):
        for l in link_distances:
            network_topologies = gen_topologies(num_network_nodes, num_comm_q=4, num_storage_q=4, link_distance=l)
            topology = network_topologies[0]
            protocols = []
            demands = [('0', str(i), f, 0.0001) for f in [0.5 + 0.05*j for j in range(8)]]
            for demand in demands:
                s, d, _, _ = demand
                protocol = get_protocol(topology, demand)
                if protocol:
                    print("Found protocol between {} and {} with fidelity {} and rate {}".format(s, d, protocol.F, protocol.R))
                    protocols.append((demand, protocol))

            action_durations = []
            for _, protocol in protocols:
                q = []
                q.append(protocol)
                while q:
                    action = q.pop(0)
                    if action.duration:
                        action_durations.append(round(action.duration, 4))
                    if hasattr(action, "protocols"):
                        q += action.protocols

            action_durations = list(set(action_durations))
            slot_sizes = sorted(list(set([0.004*i for i in range(1, 25)])))
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


from jobscheduling.protocols import schedule_dag_asap, convert_task_to_alap, shift_distillations_and_swaps


def example_schedule():
    network_topologies = gen_topologies(9, num_comm_q=1, num_storage_q=1, link_distance=5)
    grid_topology = network_topologies[2]
    demands = [('0,1', '2,1', 0.8, 1), ('1,0', '1,2', 0.8, 1)]
    taskset = []
    for demand in demands:
        protocol = get_protocol(grid_topology, demand)
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
    protocol = get_protocol(line_topology, demand)
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
                    protocol = get_protocol(line_topology, demand)
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


def throughput_vs_path_length():
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
                    protocol = get_protocol(line_topology, demand)
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


def get_wcrt_in_slots(schedule, slot_size):
    task_wcrts = defaultdict(int)
    for s, e, t in schedule:
        name_components = t.name.split("|")
        instance_name = "|".join(name_components[:2])
        task_wcrts[instance_name] = (e - t.a)*slot_size

    original_task_wcrts = defaultdict(int)
    for instance_name, wcrt in task_wcrts.items():
        original_taskname = instance_name.split("|")[0]
        original_task_wcrts[original_taskname] = max(original_task_wcrts[original_taskname], wcrt)

    return original_task_wcrts

import numpy as np

def get_start_jitter_in_slots(taskset, schedule, slot_size):
    periodic_task_starts = defaultdict(list)
    for s, e, t in schedule:
        name_components = t.name.split("|")
        original_taskname = name_components[0]
        if len(name_components) == 3:
            if name_components[2] == "0":
                periodic_task_starts[original_taskname].append(s*slot_size)
        else:
            periodic_task_starts[original_taskname].append(s*slot_size)

    periodic_task_start_jitter = defaultdict(int)
    for periodic_task in taskset:
        task_starts = periodic_task_starts[periodic_task.name]
        change = []
        for s1, s2 in zip(task_starts, task_starts[1:]):
            diff = s2 - s1 - periodic_task.p * slot_size
            change.append(diff)

        if change:
            jitter = np.var(change)
            periodic_task_start_jitter[periodic_task.name] = jitter

    return periodic_task_start_jitter


def get_taskset(num_tasks, fidelity, topology, slot_size):
    taskset = []
    num_succ = 0
    while len(taskset) < num_tasks:
        s, d = random.sample(['0', '1', '2', '3'], 2)
        rate = 10 / (2**random.choice([i for i in range(3, 7)]))
        demand = (s, d, fidelity, rate)
        try:
            logger.debug("Constructing protocol for request {}".format(demand))
            protocol = get_protocol(topology, demand)
            if protocol is None:
                logger.warning("Demand {} could not be satisfied!".format(demand))
                continue

            logger.debug("Converting protocol for request {} to task".format(demand))
            task = convert_protocol_to_task(demand, protocol, slot_size)

            logger.debug("Scheduling task for request {}".format(demand))

            scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, topology)

            latency = scheduled_task.c * slot_size
            achieved_rate = 1 / latency

            s, d, f, r = demand
            asap_dec, alap_dec, shift_dec = decoherence_times
            logger.info("Results for {}:".format(demand))
            if not correct:
                logger.error("Failed to construct valid protocol for {}".format(demand))
            elif shift_dec > asap_dec or shift_dec > alap_dec:
                logger.error("Shifted protocol has greater decoherence than ALAP or ASAP for demand {}".format(demand))
            elif achieved_rate < r:
                logger.warning("Failed to satisfy rate for {}, achieved {}".format(demand, achieved_rate))
            else:
                num_succ += 1
                logger.info(
                    "Successfully created protocol and task for demand (S={}, D={}, F={}, R={}), {}".format(*demand,
                                                                                                            num_succ))
                taskset.append(scheduled_task)

        except Exception as err:
            logger.exception("Error occurred while generating tasks: {}".format(err))

    return taskset


def sample_sim():
    center_resource_configs = [(1, 3), (2, 3), (1, 4), (2, 4)]
    end_node_resources = (1, 3)
    fidelities = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    num_tasksets = 5
    num_tasks = 40
    slot_size = 0.05
    schedulers = get_schedulers()

    all_data = {}
    for center_resources in center_resource_configs:
        resource_config_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        topology = gen_plus_topology(5, end_node_resources=end_node_resources, center_resources=center_resources,
                                     link_distance=5)
        network_resources = get_network_resources(topology)
        print("Running center config {}".format(center_resources))
        for fidelity in fidelities:
            print("Running fidelity {}".format(fidelity))
            for num_taskset in range(num_tasksets):
                print("Running taskset {}".format(num_taskset))
                taskset = get_taskset(num_tasks, fidelity, topology, slot_size)

                total_rate_dict = defaultdict(int)
                for task in taskset:
                    total_rate_dict[1 / task.p] += 1

                for scheduler_class in schedulers:
                    try:
                        scheduler = scheduler_class()
                        results_key = type(scheduler).__name__
                        print("Running scheduler {}".format(results_key))
                        running_taskset = []
                        last_succ_schedule = None

                        logger.info("Scheduling tasks with {}".format(results_key))
                        start = time.time()
                        for task in taskset:
                            # First test the taskset if it is even feasible to schedule
                            test_taskset = running_taskset + [task]
                            if check_resource_utilization(test_taskset, network_resources) == False:
                                continue

                            schedule = scheduler.schedule_tasks(running_taskset + [task], topology)
                            if schedule:
                                # Record success
                                if all([valid for _, _, valid in schedule]):
                                    running_taskset.append(task)
                                    logger.info("Running taskset length: {}".format(len(running_taskset)))
                                    last_succ_schedule = schedule
                                    for sub_taskset, sub_schedule, _ in schedule:
                                        logger.debug("Created schedule for {} demands {}, length={}".format(
                                            len(sub_taskset), [t.name for t in sub_taskset],
                                            max([slot_info[1] for slot_info in sub_schedule])))

                                else:
                                    logger.warning(
                                        "Could not add demand {} with latency {}".format(task.name, task.c * slot_size))

                        end = time.time()
                        logger.info("{} completed scheduling in {}s".format(results_key, end - start))
                        logger.info("{} scheduled {} tasks".format(results_key, len(running_taskset)))

                        rate_dict = defaultdict(int)

                        for task in running_taskset:
                            rate_dict[1 / task.p] += 1

                        logger.info("Taskset {} statistics:".format(num_taskset))
                        logger.info("Rates: ")

                        num_pairs = 0
                        hyperperiod = get_lcm_for([t.p for t in running_taskset])
                        for rate in sorted(total_rate_dict.keys()):
                            num_pairs += rate_dict[rate] * hyperperiod * rate
                            logger.info("{}: {} / {}".format(rate, rate_dict[rate], total_rate_dict[rate]))

                        total_latency = hyperperiod * slot_size
                        logger.info("Schedule generates {} pairs in {}s".format(num_pairs, total_latency))

                        network_throughput = num_pairs / total_latency
                        logger.info("Network Throughput: {} ebit/s".format(network_throughput))

                        task_wcrts = {}
                        task_jitters = {}
                        for sub_taskset, sub_schedule, _ in last_succ_schedule:
                            subtask_wcrts = get_wcrt_in_slots(sub_schedule, slot_size)
                            subtask_jitters = get_start_jitter_in_slots(running_taskset, sub_schedule, slot_size)
                            task_wcrts.update(subtask_wcrts)
                            task_jitters.update(subtask_jitters)

                        resource_config_data[results_key][fidelity]["throughput"].append(network_throughput)
                        resource_config_data[results_key][fidelity]["wcrt"].append(max(task_wcrts.values()))
                        resource_config_data[results_key][fidelity]["jitter"].append(max(task_jitters.values()))

                    except Exception as err:
                        logger.exception("Error occurred while scheduling: {}".format(err))

        all_data[center_resources] = resource_config_data

    final_data = {}
    for res_conf, res_conf_data in all_data.items():
        final_res_conf_key = str(res_conf)
        final_res_conf_data = {}
        for fidelity, fidelity_data in res_conf_data.items():
            final_fidelity_data = {}
            for sched_name, sched_data in fidelity_data.items():
                final_sched_data = {}
                for metric_name, metric_data in sched_data.items():
                    average_metric_data = sum(metric_data) / len(metric_data)
                    final_sched_data[metric_name] = average_metric_data

                final_fidelity_data[sched_name] = final_sched_data

            final_res_conf_data[fidelity] = final_fidelity_data

        final_data[final_res_conf_key] = final_res_conf_data

    import json
    json.dump(final_data, open("out.json", "w"), sort_keys=True, indent=4)
    plot_results(final_data)


def plot_results(data):
    import matplotlib.pyplot as plt

    schedulers = list(data["(1, 3)"].keys())
    for res_conf, res_conf_data in data.items():
        for metric in ["throughput", "wcrt", "jitter"]:
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
            metric_to_label = {"throughput": "Throughput", "wcrt": "Worst-case Response Time", "jitter": "Jitter"}
            metric_to_units = {"throughput": "(ebit/s)", "wcrt": "(s)", "jitter": "(s^2)"}
            ax.set_ylabel("{} {}".format(metric_to_label[metric], metric_to_units[metric]))
            ax.set_title('{} by scheduler and fidelity {}'.format(metric_to_label[metric], res_conf))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            fig.tight_layout()

            plt.show()


def main():
    num_network_nodes = 5
    num_tasksets = 1
    budget_allowances = [1*i for i in range(1)]
    network_topologies = gen_topologies(num_network_nodes, num_comm_q=1, num_storage_q=3)
    slot_size = 0.05
    demand_size = 40

    network_schedulers = get_schedulers()
    results = {}
    for topology in network_topologies:
        network_tasksets = []
        network_taskset_properties = []
        network_resources = get_network_resources(topology)

        for i in range(num_tasksets):
            logger.info("Generating taskset {}".format(i))
            taskset_properties = {}

            # Generate task sets according to some utilization characteristics and preemption budget allowances
            try:
                demands = get_network_demands(topology, demand_size)

                taskset = []
                num_succ = 0
                for demand in demands:
                    try:
                        logger.debug("Constructing protocol for request {}".format(demand))
                        protocol = get_protocol_without_rate_constraint(topology, demand)
                        if protocol is None:
                            logger.warning("Demand {} could not be satisfied!".format(demand))
                            continue


                        logger.debug("Converting protocol for request {} to task".format(demand))
                        task = convert_protocol_to_task(demand, protocol, slot_size)

                        logger.debug("Scheduling task for request {}".format(demand))

                        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, topology)

                        latency = scheduled_task.c * slot_size
                        achieved_rate = 1 / latency

                        new_rate = select_rate(achieved_rate)
                        s, d, f, r = demand
                        demand = (s, d, f, new_rate)

                        asap_dec, alap_dec, shift_dec = decoherence_times
                        logger.info("Results for {}:".format(demand))
                        if not correct:
                            logger.error("Failed to construct valid protocol for {}".format(demand))
                            import pdb
                            pdb.set_trace()
                        elif achieved_rate < r:
                            logger.warning("Failed to satisfy rate for {}, achieved {}".format(demand, achieved_rate))
                            import pdb
                            pdb.set_trace()
                        elif shift_dec > asap_dec or shift_dec > alap_dec:
                            logger.error("Shifted protocol has greater decoherence than ALAP or ASAP for demand {}".format(demand))
                            import pdb
                            pdb.set_trace()
                        else:
                            num_succ += 1
                            logger.info("Successfully created protocol and task for demand (S={}, D={}, F={}, R={}), {}".format(*demand, num_succ))
                            taskset.append(scheduled_task)

                    except Exception as err:
                        logger.exception("Error occurred while generating tasks: {}".format(err))

                logger.info("Demands: {}".format(demands))
                total_rate_dict = defaultdict(int)
                for task in taskset:
                    total_rate_dict[1 / task.p] += 1

                taskset_properties["rates"] = total_rate_dict
                network_tasksets.append(taskset)
                network_taskset_properties.append(taskset_properties)

                logger.info("Completed creating taskset {}".format(i))
                # Use all schedulers
                for scheduler_class in network_schedulers:
                    try:
                        scheduler = scheduler_class()
                        results_key = type(scheduler).__name__
                        running_taskset = []
                        last_succ_schedule = None

                        logger.info("Scheduling tasks with {}".format(results_key))
                        start = time.time()
                        for task in taskset:
                            # First test the taskset if it is even feasible to schedule
                            test_taskset = running_taskset + [task]
                            if check_resource_utilization(test_taskset, network_resources) == False:
                                continue

                            schedule = scheduler.schedule_tasks(running_taskset + [task], topology)
                            if schedule:
                                # Record success
                                if all([valid for _, _, valid in schedule]):
                                    running_taskset.append(task)
                                    logger.info("Running taskset length: {}".format(len(running_taskset)))
                                    last_succ_schedule = schedule
                                    for sub_taskset, sub_schedule, _ in schedule:
                                        logger.debug("Created schedule for {} demands {}, length={}".format(
                                            len(sub_taskset), [t.name for t in sub_taskset],
                                            max([slot_info[1] for slot_info in sub_schedule])))

                                else:
                                    logger.warning("Could not add demand {} with latency {}".format(task.name, task.c*slot_size))

                        end = time.time()
                        logger.info("{} completed scheduling in {}s".format(results_key, end - start))
                        logger.info("{} scheduled {} tasks".format(results_key, len(running_taskset)))

                        rate_dict = defaultdict(int)

                        for task in running_taskset:
                            rate_dict[1/task.p] += 1

                        logger.info("Taskset {} statistics:".format(i))
                        logger.info("Rates: ")

                        num_pairs = 0
                        hyperperiod = get_lcm_for([t.p for t in running_taskset])
                        for rate in sorted(total_rate_dict.keys()):
                            num_pairs += rate_dict[rate] * hyperperiod * rate
                            logger.info("{}: {} / {}".format(rate, rate_dict[rate], total_rate_dict[rate]))

                        total_latency = hyperperiod*slot_size
                        logger.info("Schedule generates {} pairs in {}s".format(num_pairs, total_latency))

                        network_throughput = num_pairs / total_latency
                        logger.info("Network Throughput: {} ebit/s".format(network_throughput))

                        for sub_taskset, sub_schedule, _ in last_succ_schedule:
                            sub_schedule_pairs = 0
                            for task in sub_taskset:
                                sub_schedule_pairs += hyperperiod / task.p
                            logger.info("Sub taskset {}: Num pairs {} Latency {}".format([t.name for t in sub_taskset],
                                                                                         sub_schedule_pairs,
                                                                                         slot_size*max([slot_info[1] for slot_info in sub_schedule])))
                            schedule_and_resource_timelines(sub_taskset, sub_schedule, plot_title=results_key)

                        # Data is taskset_num, number_scheduled_tasks, overall throughput, rate dict
                        # satisfied demands
                        scheduler_results = (i, len(running_taskset), network_throughput, rate_dict, [t.name for t in running_taskset])
                        results[results_key] = scheduler_results

                    except Exception as err:
                        logger.exception("Error occurred while scheduling: {}".format(err))

            except Exception as err:
                logger.exception("Unknown error occurred: {}".format(err))

        import pdb
        pdb.set_trace()

    # Plot schedulability ratio vs. utilization for each task set
    # for scheduler_type, scheduler_results in results.items():
    #     xdata = utilizations
    #     ydata = [scheduler_results[u] for u in utilizations]
    #     plt.plot(xdata, ydata, label=scheduler_type)
    #
    # plt.show()

    # Plot schedulability ratio vs. budget allowances for each task set
    for scheduler_type, scheduler_results in results.items():
        xdata = budget_allowances
        ydata = [scheduler_results[b] for b in budget_allowances]
        plt.plot(xdata, ydata, lable=scheduler_type)

    plt.show()


if __name__ == "__main__":
    main()
    # slot_size_selection()
    # throughput_vs_path_length()
    # visualize_scheduled_protocols()
    # visualize_protocol_scheduling()
    # example_schedule()
    # sample_sim()
    # import json
    # data = json.load(open("out.json"))
    # plot_results(data)
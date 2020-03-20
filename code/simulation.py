import matplotlib.pyplot as plt
import networkx as nx
import random
import time
from math import sqrt
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
from jobscheduling.visualize import draw_DAG, schedule_timeline, resource_timeline, schedule_and_resource_timelines
from math import ceil

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
        # MultipleResourceNonBlockNPRMScheduler,
    ]
    return schedulers


def get_network_demands(network_topology, num):
    _, nodeG = network_topology
    demands = []
    for num_demands in range(num):
        src, dst = random.sample(nodeG.nodes, 2)
        fidelity = round(0.6 + random.random() * (3 / 10), 3)                    # Fidelity range between F=0.6 and 1
        rate = 10 / (2**random.choice([i for i in range(5, 9)]))       # Rate range between 0.2 and 1
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


from collections import defaultdict
from intervaltree import IntervalTree, Interval


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
    link_distances = list(range(5, 55, 5))
    for i in range(1, num_network_nodes):
        for l in link_distances:
            network_topologies = gen_topologies(num_network_nodes, num_comm_q=4, num_storage_q=4, link_distance=l)
            topology = network_topologies[0]
            protocols = []
            demands = [('0', str(i), f, 0.0001) for f in [0.5 + 0.05*j for j in range(10)]]
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
                plt.plot(xdata, ydata, label=str(demand))

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
                plt.plot(xdata, ydata, label=str(demand))

            plt.legend()
            plt.autoscale()
            plt.xlabel("Slot Size (s)")
            plt.ylabel("Latency (# slots)")
            plt.title("Protocol Latency vs. Slot Size")
            plt.show()


def main():
    num_network_nodes = 8
    num_tasksets = 1
    budget_allowances = [1*i for i in range(1)]
    utilizations = [0.1*i for i in range(1, 10)]
    network_topologies = gen_topologies(num_network_nodes, num_comm_q=4, num_storage_q=4)
    slot_size = 0.05
    demand_size = 10

    network_schedulers = get_schedulers()
    results = {}
    for topology in network_topologies:
        network_tasksets = []

        for i in range(num_tasksets):
            logger.info("Generating taskset {}".format(i))

            # Generate task sets according to some utilization characteristics and preemption budget allowances
            demands = get_network_demands(topology, demand_size)

            logger.info("Demands: {}".format(demands))

            taskset = []
            num_succ = 0
            for demand in demands:
                logger.info("Constructing protocol for request {}".format(demand))
                protocol = get_protocol(topology, demand)
                if protocol is None:
                    logger.warning("Demand {} could not be satisfied!".format(demand))
                    continue

                logger.debug("Converting protocol for request {} to task".format(demand))
                task = convert_protocol_to_task(demand, protocol, slot_size)

                logger.debug("Scheduling task for request {}".format(demand))

                scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, topology)

                sink = scheduled_task.sinks[0]
                latency = (sink.a + ceil(sink.c)) * slot_size
                achieved_rate = 1 / latency

                s, d, f, r = demand
                asap_dec, alap_dec, shift_dec = decoherence_times
                logger.info("Results for {}:".format(demand))
                if not correct:
                    logger.error("Failed to construct valid protocol for {}".format(demand))
                    import pdb
                    pdb.set_trace()

                elif shift_dec > asap_dec or shift_dec > alap_dec:
                    logger.error("Shifted protocol has greater decoherence than ALAP or ASAP")
                    import pdb
                    pdb.set_trace()

                else:
                    num_succ += 1
                    logger.info("Successfully created protocol and task for demand (S={}, D={}, F={}, R={}), {}".format(*demand, num_succ))
                    taskset.append(scheduled_task)

            logger.info("Created taskset {}".format([t.name for t in taskset]))
            network_tasksets.append(taskset)
            # Use all schedulers
            for scheduler_class in network_schedulers:
                scheduler = scheduler_class()
                results_key = type(scheduler).__name__
                running_taskset = []
                last_succ_schedule = None
                start = time.time()
                for task in taskset:
                    logger.debug("Scheduling tasks with {}".format(results_key))
                    schedule = scheduler.schedule_tasks(running_taskset + [task], topology)
                    if schedule:
                        # Record success
                        if all([valid for _, _, valid in schedule]):
                            running_taskset.append(task)
                            print(len(running_taskset))
                            last_succ_schedule = schedule
                            for sub_taskset, sub_schedule, valid in schedule:
                                logger.debug("Created schedule for sub_taskset size {}, valid={}, length={}".format(
                                    len(sub_taskset), valid, max([slot_info[1] for slot_info in sub_schedule])))

                        else:
                            print("Could not add task")

                    else:
                        logger.info("Failed to create a schedule for taskset")
                        import pdb
                        pdb.set_trace()

                end = time.time()
                logger.info("{} completed scheduling in {}s".format(results_key, end - start))

                scheduler_results = len(running_taskset)
                results[results_key] = scheduler_results
                logger.info("{} scheduled {} tasks".format(results_key, scheduler_results))
                logger.info("Taskset {} statistics:".format(i))
                logger.info("Rates: ")
                rate_dict = defaultdict(int)
                total_rate_dict = defaultdict(int)
                num_pairs = 0

                for task in taskset:
                    total_rate_dict[1/task.p] += 1
                    if task in running_taskset:
                        rate_dict[1/task.p] += 1

                hyperperiod = get_lcm_for([t.p for t in running_taskset])
                for rate in sorted(total_rate_dict.keys()):
                    num_pairs += rate_dict[rate] * hyperperiod * rate
                    logger.info("{}: {} / {}".format(rate, rate_dict[rate], total_rate_dict[rate]))
                throughput = sum([num_pairs / (slot_size * max([slot_info[1] for slot_info in sub_schedule])) for _, sub_schedule, _ in last_succ_schedule])
                logger.info("Network Throughput: {} ebit/s".format(throughput))
                for sub_taskset, sub_schedule, _ in last_succ_schedule:
                    schedule_and_resource_timelines(sub_taskset, sub_schedule, plot_title=results_key)

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

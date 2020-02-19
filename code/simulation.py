import matplotlib.pyplot as plt
import networkx as nx
import random
import time
from math import sqrt
from device_characteristics.nv_links import load_link_data
from jobscheduling.log import LSLogger
from jobscheduling.protocols import convert_protocol_to_task, create_protocol, LinkProtocol, DistillationProtocol, SwapProtocol, schedule_dag_for_resources
from jobscheduling.scheduler import MultipleResourceOptimalBlockScheduler, MultipleResourceBlockNPEDFScheduler, MultipleResourceBlockCEDFScheduler, MultipleResourceNonBlockNPEDFScheduler, PreemptionBudgetScheduler, pretty_print_schedule
from jobscheduling.visualize import draw_DAG


logger = LSLogger()


def get_dimensions(n):
    divisors = []
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
            divisors.append(currentDiv+1)

    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i]-sqrt(n)))
    wIndex = hIndex + 1

    return divisors[hIndex], divisors[wIndex]


def gen_topologies(n, num_comm_q=2, num_storage_q=2):
    d_to_cap = load_link_data()
    link_distance = 5
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
            for j in range(4):
                for k in range(4):
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
            for j in range(4):
                for k in range(4):
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
            for j in range(4):
                for k in range(4):
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
                for k in range(4):
                    for l in range(4):
                        gridGcq.add_edge("{},{}-C{}".format(i, j-1, k), "{},{}-C{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance)
            # Connect left
            if i > 0:
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i - 1, j), capabilities=link_capability,
                               weight=link_distance)
                for k in range(4):
                    for l in range(4):
                        gridGcq.add_edge("{},{}-C{}".format(i-1, j, k), "{},{}-C{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance)

    return [(demoGcq, demoG), (lineGcq, lineG), (ringGcq, ringG), (gridGcq, gridG)]


def get_schedulers():
    schedulers = [
        # MultipleResourceOptimalBlockScheduler,
        MultipleResourceBlockNPEDFScheduler,
        MultipleResourceNonBlockNPEDFScheduler,
        MultipleResourceBlockCEDFScheduler
    ]
    return schedulers


def get_network_demands(network_topology, num):
    _, nodeG = network_topology
    demands = []
    for num_demands in range(num):
        src, dst = random.sample(nodeG.nodes, 2)
        fidelity = round(0.75 + random.random() / 4, 3)  # Fidelity range between F=0.75 and 1
        rate = round(0.2 + random.choice([0.1*i for i in range(1, 9)]), 3)       # Rate range between 0.2 and 1
        demands.append((src, dst, fidelity, rate))
    return demands


def get_protocol(network_topology, demand):
    s, d, f, r = demand
    _, nodeG = network_topology
    path = nx.shortest_path(G=nodeG, source=s, target=d, weight="weight")
    protocol = create_protocol(path, nodeG, f, r)
    if protocol:
        return protocol
    else:
        logger.warning("Demand (S={}, D={}, F={}, R={}) could not be satisfied!".format(s, d, f, r))
        return None



def main():
    num_network_nodes = 10
    num_tasksets = 1
    budget_allowances = [1*i for i in range(1)]
    network_topologies = gen_topologies(num_network_nodes)

    network_schedulers = get_schedulers()
    schedule_validator = PreemptionBudgetScheduler()
    results = {}

    for topology in network_topologies:
        network_tasksets = []

        for i in range(num_tasksets):
            logger.info("Generating taskset {}".format(i))
            # Generate task sets according to some utilization characteristics and preemption budget allowances
            demands = get_network_demands(topology, 10)

            logger.info("Demands: {}".format(demands))

            taskset = []
            for demand in demands:
                logger.debug("Constructing protocol for request {}".format(demand))
                protocol = get_protocol(topology, demand)
                if protocol is None:
                    continue

                logger.debug("Converting protocol for request {} to task".format(demand))
                task = convert_protocol_to_task(demand, protocol)

                logger.debug("Scheduling task for request {}".format(demand))
                scheduled_task = schedule_dag_for_resources(task, topology)

                logger.info("Created protocol and task for demand (S={}, D={}, F={}, R={})".format(*demand))
                taskset.append(scheduled_task)

            logger.info("Created taskset {}".format([t.name for t in taskset]))
            network_tasksets.append(taskset)

        # Use all schedulers
        for scheduler_class in network_schedulers:
            scheduler = scheduler_class()
            results_key = str(type(scheduler))
            scheduler_results = []

            # Run scheduler on all task sets
            for i in range(num_tasksets):
                taskset = network_tasksets[i]
                logger.info("Scheduling tasks with {}".format(type(scheduler).__name__))
                start = time.time()
                schedule = scheduler.schedule_tasks(taskset)
                end = time.time()
                logger.info("Completed scheduling in {}s".format(end - start))
                if schedule:
                    for sub_taskset, sub_schedule, valid in schedule:
                        logger.info("Created schedule for sub_taskset {}, valid={}".format([t.name for t in sub_taskset], valid))

                    # Record success
                    scheduler_results.append(all([valid for _, _, valid in schedule]) if schedule else False)

                else:
                    logger.info("Failed to create a schedule for taskset")

            results[results_key] = scheduler_results

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

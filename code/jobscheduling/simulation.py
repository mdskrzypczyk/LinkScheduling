import matplotlib.pyplot as plt
import random
from collections import defaultdict
import networkx as nx
from protocols import create_protocol
from task import DAGResourceSubTask, ResourceDAGTask
from esss import LinkProtocol, DistillationProtocol, SwapProtocol
from visualize import draw_DAG
from nv_links import load_link_data

def gen_topologies(n):
    d_to_cap = load_link_data()
    # Line
    lineGcq = nx.Graph()
    lineG = nx.Graph()
    for i in range(n):
        comm_qs = []
        for num_comm_q in range(1):
            comm_q_id = "{}-{}".format(i, num_comm_q)
            comm_qs.append(comm_q_id)
        lineGcq.add_nodes_from(comm_qs)
        lineG.add_node("{}".format(i))
        if i > 0:
            prev_node_id = i - 1
            for j in range(1):
                for k in range(1):
                    lineGcq.add_edge("{}-{}".format(prev_node_id, j), "{}-{}".format(i, k))
            lineG.add_edge("{}".format(prev_node_id), "{}".format(i),
                           capabilities=d_to_cap['5'])

    # Ring
    ringGcq = nx.Graph()
    ringG = nx.Graph()
    for i in range(n):
        comm_qs = []
        for num_comm_q in range(1):
            comm_q_id = "{}-{}".format(i, num_comm_q)
            comm_qs.append(comm_q_id)
        ringGcq.add_nodes_from(comm_qs)
        ringG.add_node("{}".format(i))
        if i > 0:
            prev_node_id = i - 1
            for j in range(1):
                for k in range(1):
                    ringGcq.add_edge("{}-{}".format(prev_node_id, j), "{}-{}".format(i, k))
            ringG.add_edge("{}".format(prev_node_id), "{}".format(i))


    for j in range(1):
        for k in range(1):
            ringGcq.add_edge("{}-{}".format(0, j), "{}-{}".format(n-1, k))

    # Grid
    # w = n // sqrt(n)
    # h = n // w
    # gridGcq = nx.Graph()
    # gridG = nx.Graph()
    # for i in range(w):
    #     for j in range(h):
    #         comm_qs = []
    #         for num_comm_q in range(1):
    #             comm_q_id = "Node{},{}-CQ{}".format(i, j, num_comm_q)
    #             comm_qs.append(comm_q_id)
    #         ringGcq.add_nodes_from(comm_qs)
    #
    #         # Connect upward
    #         if j > 0:
    #             for k in range(1):
    #                 for l in range(1):
    #                     ringGcq.add_edge("Node{},{}-CQ{}".format(i, j-1, k), "Node{},{}-CQ{}".format(i, j, l))
    #         # Connect left
    #         if i > 0:
    #             for k in range(1):
    #                 for l in range(1):
    #                     ringGcq.add_edge("Node{},{}-CQ{}".format(i-1, j, k), "Node{},{}-CQ{}".format(i, j, l))
    #
    #         gridG.add_node("Node{},{}".format(i, j))

    return [(lineGcq, lineG), (ringGcq, ringG)]


def get_schedulers():
    schedulers = [
        MultipleResourceEDFScheduler,
        MultipleResourceBlockEDFScheduler,
        MultipleResourceNPEDFScheduler,
        MultipleResourceBlockNPEDFScheduler,
        MultipleResourceBlockCEDFScheduler,
        MultipleResourceBlockEDFLBFScheduler
    ]
    return schedulers


def get_network_demands(network_topology):
    _, nodeG = network_topology
    demands = []
    for num_demands in range(1):
        src, dst = ['0', '3']
        fidelity = 0.9
        rate = 1 / random.sample(list(range(50, 100)), 1)[0]
        demands.append((src, dst, fidelity, rate))
    return demands


def get_protocols(network_topology, demands):
    demands_to_protocols = {}
    _, nodeG = network_topology
    for s, d, f, r in demands:
        path = nx.shortest_path(nodeG, s, d)
        protocol = create_protocol(path, nodeG, f, r)
        demands_to_protocols[(s, d, f, r)] = protocol
    return demands_to_protocols


def print_protocol(protocol):
    q = [protocol]
    while q:
        p = q.pop(0)
        print(p.name, p.F, p.R)
        if type(p) == SwapProtocol or type(p) == DistillationProtocol:
            q += p.protocols


def convert_protocol_to_task(protocol):
    q = [(protocol, None)]
    tasks = []
    labels = {
        LinkProtocol.__name__: 0,
        DistillationProtocol.__name__: 0,
        SwapProtocol.__name__: 0
    }
    while q:
        protocol_action, child_task = q.pop(0)

        name = protocol_action.name
        suffix = name.split(';')[1:]
        name = ';'.join([type(protocol_action).__name__[0]] + [str(labels[type(protocol_action).__name__])] + suffix)
        labels[type(protocol_action).__name__] += 1

        resources = protocol_action.nodes
        dagtask = DAGResourceSubTask(name=name, children=child_task, resources=resources)
        tasks.append(dagtask)

        if child_task:
            child_task[0].add_parent(dagtask)

        if type(protocol_action) in [DistillationProtocol, SwapProtocol]:
            for action in protocol_action.protocols:
                q.append((action, [dagtask]))

    main_dag_task = ResourceDAGTask(name="???", tasks=tasks)

    return main_dag_task


def main():
    num_network_nodes = 4
    num_tasksets = 1
    utilizations = [0.1*i for i in range(1, 11)]           # Utilizations in increments of 0.1
    budget_allowances = [1*i for i in range(1, 11)]
    network_topologies = gen_topologies(num_network_nodes)
    line, ring = network_topologies
    protocols = get_protocols(line, get_network_demands(line))
    task = convert_protocol_to_task(list(protocols.values())[0])
    draw_DAG(task)
    import pdb
    pdb.set_trace()

    network_schedulers = get_schedulers()

    schedule_validator = MultipleResourceBlockEDFLBFScheduler.check_feasible

    results = {}

    for topology in network_topologies:
        network_tasksets = defaultdict(list)

        for u in utilizations:
            for i in range(num_tasksets):
                # Generate task sets according to some utilization characteristics and preemption budget allowances
                # 1) Select S/D pairs in the network topology
                demands = get_network_demands(topology, u)

                # 2) Select protocol for each S/D pair
                protocols = get_protocols(topology, demands)

                # 3) Convert to task representation
                taskset = []
                for protocol in protocols:
                    taskset.append(convert_protocol_to_task(protocol))

                network_tasksets[u].append(taskset)

        # Use all schedulers
        for scheduler in network_schedulers:
            results_key = str(type(scheduler))
            scheduler_results = defaultdict(int)

            for u in utilizations:
                # Run scheduler on all task sets
                for taskset in network_tasksets[u]:
                    schedule, _ = scheduler.schedule_tasks(taskset)
                    valid = schedule_validator(schedule, taskset)

                    # Record success
                    if valid:
                        scheduler_results[u] += 1

                scheduler_results[u] /= len(network_tasksets[u])

            results[results_key] = scheduler_results

    # Plot schedulability ratio vs. utilization for each task set
    for scheduler_type, scheduler_results in results.items():
        xdata = utilizations
        ydata = [scheduler_results[u] for u in utilizations]
        plt.plot(xdata, ydata, label=scheduler_type)

    plt.show()

    # Plot schedulability ratio vs. budget allowances for each task set
    for scheduler_type, scheduler_results in results.items():
        xdata = budget_allowances
        ydata = [scheduler_results[b] for b in budget_allowances]
        plt.plot(xdata, ydata, lable=scheduler_type)

    plt.show()


if __name__ == "__main__":
    main()

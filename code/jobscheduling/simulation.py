import matplotlib.pyplot as plt
import random
from collections import defaultdict
import networkx as nx
from protocols import create_protocol
from task import DAGResourceSubTask, ResourceDAGTask
from esss import LinkProtocol, DistillationProtocol, SwapProtocol
from visualize import draw_DAG
from nv_links import load_link_data
from math import sqrt, ceil


def get_dimensions(n):
    tempSqrt = sqrt(n)
    divisors = []
    currentDiv = 1
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
            divisors.append(currentDiv+1)

    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i]-sqrt(n)))
    wIndex = hIndex + 1

    return divisors[hIndex], divisors[wIndex]


def gen_topologies(n):
    d_to_cap = load_link_data()
    link_distance = 5
    link_capability = d_to_cap[str(link_distance)]
    # Line
    lineGcq = nx.Graph()
    lineG = nx.Graph()
    for i in range(n):
        comm_qs = []
        storage_qs = []
        for num_comm_q in range(3):
            comm_q_id = "{}-C{}".format(i, num_comm_q)
            comm_qs.append(comm_q_id)
        for num_storage_q in range(1):
            storage_q_id = "{}-S{}"
            storage_qs.append(storage_q_id)
        lineGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        lineG.add_node("{}".format(i), comm_qs=comm_qs)
        if i > 0:
            prev_node_id = i - 1
            for j in range(1):
                for k in range(1):
                    lineGcq.add_edge("{}-{}".format(prev_node_id, j), "{}-{}".format(i, k))
            lineG.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability, weight=link_distance)

    # Ring
    ringGcq = nx.Graph()
    ringG = nx.Graph()
    for i in range(n):
        comm_qs = []
        for num_comm_q in range(1):
            comm_q_id = "{}-{}".format(i, num_comm_q)
            comm_qs.append(comm_q_id)
        for num_storage_q in range(1):
            storage_q_id = "{}-S{}"
            storage_qs.append(storage_q_id)
        ringGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        ringG.add_node("{}".format(i), comm_qs=comm_qs)
        if i > 0:
            prev_node_id = i - 1
            for j in range(1):
                for k in range(1):
                    ringGcq.add_edge("{}-{}".format(prev_node_id, j), "{}-{}".format(i, k))
            ringG.add_edge("{}".format(prev_node_id), "{}".format(i), capabilities=link_capability, weight=link_distance)

    ringG.add_edge("{}".format(0), "{}".format(n-1), capabilities=link_capability, weight=link_distance)

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
            for num_comm_q in range(1):
                comm_q_id = "{},{}-{}".format(i, j, num_comm_q)
                comm_qs.append(comm_q_id)
            for num_storage_q in range(1):
                storage_q_id = "{}-S{}"
                storage_qs.append(storage_q_id)
            gridGcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
            gridG.add_node("{},{}".format(i, j), comm_qs=comm_qs)

            # Connect upward
            if j > 0:
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i, j-1), capabilities=link_capability,
                               weight=link_distance)
                for k in range(1):
                    for l in range(1):
                        gridGcq.add_edge("{},{}-{}".format(i, j-1, k), "{},{}-{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance,)
            # Connect left
            if i > 0:
                gridG.add_edge("{},{}".format(i, j), "{},{}".format(i - 1, j), capabilities=link_capability,
                               weight=link_distance)
                for k in range(1):
                    for l in range(1):
                        gridGcq.add_edge("{},{}-{}".format(i-1, j, k), "{},{}-{}".format(i, j, l),
                                         capabilities=link_capability, weight=link_distance,)

    return [(lineGcq, lineG), (ringGcq, ringG), (gridGcq, gridG)]


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


def get_network_demands(network_topology, num):
    _, nodeG = network_topology
    demands = []
    for num_demands in range(num):
        src, dst = ['0', '3'] # random.sample(nodeG.nodes, 2)
        fidelity = 0.7577304591440375 # 0.5 + random.random() / 2
        rate = 1.6666666666666665 # 1 / random.sample(list(range(5, 10)), 1)[0] * 10
        demands.append((src, dst, fidelity, rate))
    return demands


def get_protocols(network_topology, demands):
    demands_to_protocols = {}
    _, nodeG = network_topology
    for s, d, f, r in demands:
        path = nx.shortest_path(G=nodeG, source=s, target=d, weight="weight")
        print("Creating protocol for request {}".format((s, d, f, r)))
        protocol = create_protocol(path, nodeG, f, r)
        if protocol:
            demands_to_protocols[(s, d, f, r)] = protocol
        else:
            print("Demand {} could not be satisfied!".format((s, d, f, r)))
    return demands_to_protocols


def print_protocol(protocol):
    q = [protocol]
    while q:
        p = q.pop(0)
        print(p.name, p.F, p.R)
        if type(p) == SwapProtocol or type(p) == DistillationProtocol:
            q += p.protocols


def convert_protocol_to_task(request, protocol):
    tasks = []
    labels = {
        LinkProtocol.__name__: 0,
        DistillationProtocol.__name__: 0,
        SwapProtocol.__name__: 0
    }

    stack = []
    protocol_action, child_action = (protocol, None)
    parent_tasks = defaultdict(list)
    last_action = None
    while stack or protocol_action != None:
        if protocol_action is not None:
            stack.append((protocol_action, child_action))

            if type(protocol_action) != LinkProtocol:
                left_protocol_action = protocol_action.protocols[0]
                child_action = protocol_action
                protocol_action = left_protocol_action

            else:
                protocol_action = None
                child_action = None

        else:
            peek_protocol_action, peek_child_action = stack[-1]
            right_protocol_action = None if type(peek_protocol_action) == LinkProtocol else peek_protocol_action.protocols[1]
            if right_protocol_action is not None and last_action != right_protocol_action:
                protocol_action = right_protocol_action
                child_action = peek_protocol_action

            else:
                name = peek_protocol_action.name
                suffix = name.split(';')[1:]
                name = ';'.join([type(peek_protocol_action).__name__[0]] + [str(labels[type(peek_protocol_action).__name__])] + suffix)
                labels[type(peek_protocol_action).__name__] += 1
                print(name)

                resources = peek_protocol_action.nodes

                dagtask = DAGResourceSubTask(name=name, c=peek_protocol_action.duration, parents=parent_tasks[peek_protocol_action.name],
                                             resources=resources)

                for parent in parent_tasks[peek_protocol_action.name]:
                    parent.add_child(dagtask)

                tasks.append(dagtask)

                if peek_child_action:
                    parent_tasks[peek_child_action.name].append(dagtask)

                last_action, _ = stack.pop()

    source, dest, fidelity, rate = request
    main_dag_task = ResourceDAGTask(name="{},{},{},{}".format(source, dest, fidelity, rate), tasks=tasks, d=1 / rate)

    return main_dag_task


def schedule_dag_for_resources(dagtask, topology):
    print("Scheduling task ASAP")
    comm_q_topology, node_topology = topology
    [sink] = dagtask.sinks
    nodes = dagtask.resources
    node_resources = dict([(n, node_topology.nodes[n]["comm_qs"]) for n in nodes])
    resource_schedules = defaultdict(list)
    resource_states = defaultdict(int)
    stack = []
    task = sink
    last_task = None
    while stack or task:
        if task is not None:
            stack.append(task)
            task = None if not task.parents else sorted(task.parents, key=lambda pt: pt.c)[1]

        else:
            peek_task = stack[-1]
            right_task = None if not peek_task.parents else sorted(peek_task.parents, key=lambda pt: pt.c)[0]
            if right_task is not None and last_task != right_task:
                task = right_task

            else:
                if peek_task.name[0] == "L":
                    possible_task_resources = [list(filter(lambda r: resource_states[r] == 0, node_resources[n])) for n in peek_task.resources]
                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules)
                    for r in used_resources:
                        resource_states[r] = 1

                elif peek_task.name[0] == "S":
                    [task_node] = peek_task.resources
                    possible_task_resources = []
                    for pt in peek_task.parents:
                        if pt.name[0] == "S":
                            q = [pt]

                            while q:
                                search_task = q.pop(0)
                                for p in search_task.parents:
                                    if p.name[0] == "S":
                                        q.append(p)
                                    else:
                                        possible_task_resources += [[r] for sp in search_task.parents for r in sp.resources if r in node_resources[task_node] and resource_states[r] == 1]
                        else:
                            possible_task_resources += [[r] for pt in peek_task.parents for r in pt.resources if r in node_resources[task_node] and resource_states[r] == 1]


                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules)
                    for r in used_resources:
                        resource_states[r] = 0

                else:
                    # Find the resources that are being distilled
                    possible_task_resources = []
                    task_nodes = peek_task.resources
                    for pt in peek_task.parents:
                        if pt.name[0] == "S":
                            q = [pt]

                            while q:
                                search_task = q.pop(0)
                                for p in search_task.parents:
                                    if p.name[0] == "S":
                                        q.append(p)
                                    else:
                                        possible_task_resources += [[r] for tn in task_nodes for sp in search_task.parents for r in sp.resources if r in node_resources[tn] and resource_states[r] == 1]
                        elif pt.name[0] == "D":
                            possible_task_resources += [[list(sorted(pt.resources))[1]]] + [[list(sorted(pt.resources))[3]]]
                        else:
                            possible_task_resources += [[r] for tn in task_nodes for r in pt.resources if r in node_resources[tn]]

                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules)

                    nrs = [list(sorted(filter(lambda r: r in node_resources[tn], used_resources))) for tn in task_nodes]
                    for nr in nrs:
                        resource_states[nr[0]] = 0

                    for r in used_resources:
                        schedule = resource_schedules[r]
                        last_occupied_time = schedule[-2]
                        resource_schedules[r] = list(sorted(set(schedule + list(range(last_occupied_time + 1, peek_task.a)))))

                last_task = stack.pop()

    draw_DAG(dagtask)
    import pdb
    pdb.set_trace()
    print("Scheduling task ALAP")
    convert_task_to_alap(dagtask)
    return dagtask

import itertools


def to_ranges(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def schedule_task_asap(task, task_resources, resource_schedules):
    earliest_start = max([0] + [p.a + ceil(p.c) for p in task.parents])
    earliest_possible_start = float('inf')
    earliest_resources = None
    for resource_set in list(itertools.product(*task_resources)):
        schedule_set = [resource_schedules[r] for r in resource_set]
        possible_start = get_earliest_start_for_resources(earliest_start, task, schedule_set)

        if possible_start < earliest_possible_start:
            earliest_possible_start = possible_start
            earliest_resources = resource_set

        if possible_start == earliest_start:
            break

    slots = list(range(earliest_possible_start, earliest_possible_start + ceil(task.c)))
    for r in list(set(earliest_resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    task.a = slots[0]
    task.resources = list(set(earliest_resources))
    print("Scheduled {} with resources {} at t={}".format(task.name, task.resources, task.a))
    return list(set(earliest_resources))


def get_earliest_start_for_resources(earliest, task, resource_schedules):
    occupied_slots = set()
    for rs in resource_schedules:
        occupied_slots |= set(rs)
    occupied_slots = list(filter(lambda s: s >= earliest, list(sorted(occupied_slots))))
    if not occupied_slots or occupied_slots[0] >= earliest + task.c:
        return earliest
    else:
        occupied_ranges = list(to_ranges(occupied_slots))
        for (s1, e1), (s2, e2) in zip(occupied_ranges, occupied_ranges[1:]):
            if s2 - e1 >= task.c:
                return e1

        return occupied_ranges[-1][1] + 1


def convert_task_to_alap(dagtask):
    stack = dagtask.sinks
    resource_schedules = defaultdict(list)
    resource_states = defaultdict(int)
    while stack:
        task = stack.pop()
        schedule_task_alap(task, resource_schedules)
        stack += list(sorted(task.parents, key=lambda task: task.a))


def schedule_task_alap(task, resource_schedules):
    latest = min([ct.a - ceil(task.c) for ct in task.children] + [task.a])
    schedule_set = [resource_schedules[r] for r in task.resources]
    start = get_latest_slot_for_resources(latest, task, schedule_set)

    slots = list(range(start, start + ceil(task.c)))
    for r in list(set(task.resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    if slots[0] != task.a:
        print("Moved task {} from t={} to t={}".format(task.name, task.a, slots[0]))
    task.a = slots[0]
    print("Scheduled {} with resources {} at t={}".format(task.name, task.resources, task.a))


def get_latest_slot_for_resources(latest, task, schedule_set):
    occupied_slots = set()
    for rs in schedule_set:
        occupied_slots |= set(rs)

    occupied_slots = list(filter(lambda s: s <= latest, list(sorted(occupied_slots))))
    if not occupied_slots or occupied_slots[-1] < latest:
        return latest
    else:
        occupied_ranges = list(reversed(list(to_ranges(occupied_slots))))
        for (s1, e1), (s2, e2) in zip(occupied_ranges, occupied_ranges[1:]):
            if s1 - e2 >= task.c:
                return e2 + 1

        return occupied_ranges[-1][0] - ceil(task.c)


def main():
    num_network_nodes = 4
    num_tasksets = 1
    utilizations = [0.1*i for i in range(1, 11)]           # Utilizations in increments of 0.1
    budget_allowances = [1*i for i in range(1, 11)]
    network_topologies = gen_topologies(num_network_nodes)

    results = {}

    for topology in network_topologies:
        network_tasksets = defaultdict(list)

        for u in utilizations:
            for i in range(num_tasksets):
                # Generate task sets according to some utilization characteristics and preemption budget allowances
                # 1) Select S/D pairs in the network topology
                demands = get_network_demands(topology, 10)

                # 2) Select protocol for each S/D pair
                protocols = get_protocols(topology, demands)

                # 3) Convert to task representation
                taskset = []
                for request, protocol in protocols.items():
                    print("Converting protocol for request {} to task".format(request))
                    task = convert_protocol_to_task(request, protocol)
                    print("Scheduling task for request {}".format(request))
                    scheduled_task = schedule_dag_for_resources(task, topology)
                    taskset.append(scheduled_task)
                    draw_DAG(scheduled_task)
                    import pdb
                    pdb.set_trace()

                network_tasksets[u].append(taskset)

        import pdb
        pdb.set_trace()

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

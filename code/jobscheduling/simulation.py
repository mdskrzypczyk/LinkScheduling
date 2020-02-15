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
from scheduler import MultipleResourceOptimalBlockScheduler, MultipleResourceBlockNPEDFScheduler, MultipleResourceBlockCEDFScheduler, pretty_print_schedule


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
        for num_comm_q in range(4):
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
    # schedulers = [
    #     MultipleResourceEDFScheduler,
    #     MultipleResourceBlockEDFScheduler,
    #     MultipleResourceNPEDFScheduler,
    #     MultipleResourceBlockNPEDFScheduler,
    #     MultipleResourceBlockCEDFScheduler,
    #     MultipleResourceBlockEDFLBFScheduler
    # ]
    schedulers = [
        # MultipleResourceOptimalBlockScheduler,
        MultipleResourceBlockNPEDFScheduler,
        MultipleResourceBlockCEDFScheduler
    ]
    return schedulers


def get_network_demands(network_topology, num):
    _, nodeG = network_topology
    demands = []
    for num_demands in range(num):
        src, dst = random.sample(nodeG.nodes, 2)
        fidelity = 0.5 + random.random() / 2
        rate = 1 / random.sample(list(range(5, 10)), 1)[0] * 10
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
    main_dag_task = ResourceDAGTask(name="{},{},{},{}".format(source, dest, fidelity, rate), tasks=tasks, d=ceil(1 / rate))

    return main_dag_task


def schedule_dag_for_resources(dagtask, topology):
    print("Scheduling task ASAP")
    comm_q_topology, node_topology = topology
    [sink] = dagtask.sinks
    nodes = dagtask.resources
    node_resources = dict([(n, node_topology.nodes[n]["comm_qs"]) for n in nodes])
    resource_schedules = defaultdict(list)
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
                    possible_task_resources = [node_resources[n] for n in peek_task.resources]
                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules)

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
                                    elif p.name[0] == "D":
                                        possible_task_resources += [[r] for r in list(sorted(p.resources))[1:4:2] if r in node_resources[task_node]]
                                    else:
                                        possible_task_resources += [[r] for r in p.resources if r in node_resources[task_node]]
                        elif pt.name[0] == "D":
                            possible_task_resources += [[r] for r in list(sorted(pt.resources))[1:4:2] if r in node_resources[task_node]]
                        else:
                            possible_task_resources += [[r] for r in pt.resources if r in node_resources[task_node]]


                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules)

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
                                    elif p.name[0] == "D":
                                        possible_task_resources += [[r] for r in list(sorted(p.resources))[1:4:2] for tn in task_nodes if r in node_resources[tn]]
                                    else:
                                        possible_task_resources += [[r] for tn in task_nodes for sp in search_task.parents for r in sp.resources if r in node_resources[tn]]# and resource_states[r] == 1]
                        elif pt.name[0] == "D":
                            possible_task_resources += [[r] for r in list(sorted(pt.resources))[1:4:2]]
                        else:
                            possible_task_resources += [[r] for tn in task_nodes for r in pt.resources if r in node_resources[tn]]

                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules)

                last_task = stack.pop()

    sink_task = dagtask.sinks[0]
    asap_duration = sink_task.a + ceil(sink_task.c)
    asap_decoherence = get_schedule_decoherence(resource_schedules, asap_duration)
    print("Scheduling task ALAP")
    convert_task_to_alap(dagtask)
    print("ASAP Schedule latency {} total decoherence {}".format(asap_duration, asap_decoherence))
    return dagtask


def get_schedule_decoherence(resource_schedules, completion_time):
    total_decoherence = 0
    for r in resource_schedules.keys():
        rs = resource_schedules[r]
        for (s1, t1), (s2, t2) in zip(rs, rs[1:]):
            if t1.name[0] == "L" or t1.name[0] == "D":
                total_decoherence += (s2 - ceil(t1.a) - s1)

        if rs:
            s, t = rs[-1]
            if t.name[0] == "L" or t.name[0] == "D":
                total_decoherence += (completion_time - s - ceil(t.c))

    return total_decoherence

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

        possible_start = get_earliest_start_for_resources(earliest_start, task, resource_set, resource_schedules)

        if earliest_start <= possible_start < earliest_possible_start:
            earliest_possible_start = possible_start
            earliest_resources = resource_set

        if possible_start == earliest_start:
            break

    slots = []
    for slot in range(earliest_possible_start, earliest_possible_start + ceil(task.c)):
        slots.append((slot, task))
    for r in list(set(earliest_resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    task.a = slots[0][0]
    task.resources = list(sorted(set(earliest_resources)))
    print("Scheduled {} with resources {} at t={}".format(task.name, task.resources, task.a))
    return list(set(earliest_resources))


def get_earliest_start_for_resources(earliest, task, resource_set, resource_schedules):
    occupied_slots = defaultdict(list)
    for r in resource_set:
        rs = resource_schedules[r]
        for slot, t in rs:
            occupied_slots[slot].append(t)

    if not occupied_slots:
        return earliest

    else:
        occupied_ranges = list(to_ranges(occupied_slots.keys()))
        _, e = occupied_ranges[-1]
        last_tasks = []
        for r in resource_set:
            rs = resource_schedules[r]
            filtered_schedule = list(sorted(filter(lambda slot_info: slot_info[0] <= e, rs)))
            if filtered_schedule:
                last_task_slot, last_task = filtered_schedule[-1]
                last_tasks.append(last_task)

        if not (task.name[0] == "L" and any([task_locks_resource(t, resource_set) for t in last_tasks])):
            if e >= earliest:
                return occupied_ranges[-1][1] + 1
            else:
                return earliest
        else:
            return float('inf')


def task_locks_resource(task, resources):
    link_lock = (task.name[0] == "L" and any([r in task.resources for r in resources]))
    distill_lock = (task.name[0] == "D" and any([r in list(sorted(task.resources))[1:4:2] for r in resources]))
    return link_lock or distill_lock


def convert_task_to_alap(dagtask):
    # Last task doesn't move
    queue = list(sorted(dagtask.subtasks, key=lambda task: -task.a))
    resource_schedules = defaultdict(list)
    last_task = queue.pop(0)
    for r in last_task.resources:
        resource_schedules[r] = [(s, last_task) for s in range(last_task.a, last_task.a + ceil(last_task.c))]

    while queue:
        task = queue.pop(0)

        # Find the latest t <= child.a s.t. all resources available
        schedule_task_alap(task, resource_schedules)

    earliest = min([task.a for task in dagtask.subtasks])
    if earliest != 0:
        for task in dagtask.subtasks:
            task.a -= earliest
        for r in resource_schedules.keys():
            schedule = resource_schedules[r]
            resource_schedules[r] = [(s - earliest, t) for s, t in schedule]

    sink_task = dagtask.sinks[0]
    alap_latency = sink_task.a + ceil(sink_task.c)
    alap_decoherence = get_schedule_decoherence(resource_schedules, alap_latency)
    print("ALAP Schedule latency {} total decoherence {}".format(alap_latency, alap_decoherence))


def schedule_task_alap(task, resource_schedules):
    possible = [task.a]
    child_starts = [ct.a - ceil(task.c) for ct in task.children if set(task.resources) & set(ct.resources)]
    if child_starts:
        possible += [min(child_starts)]
    latest = max(possible)

    slots = [(s, task) for s in range(latest, latest + ceil(task.c))]
    for r in list(set(task.resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    if slots[0][0] != task.a:
        print("Moved task {} from t={} to t={}".format(task.name, task.a, slots[0][0]))
    task.a = slots[0][0]
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

    network_schedulers = get_schedulers()
    # schedule_validator = MultipleResourceBlockNPEDFScheduler.check_feasible
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

                network_tasksets[u].append(taskset)


        # Use all schedulers
        for scheduler_class in network_schedulers:
            scheduler = scheduler_class()
            results_key = str(type(scheduler))
            scheduler_results = defaultdict(int)

            for u in utilizations:
                # Run scheduler on all task sets
                for taskset in network_tasksets[u]:
                    schedule = scheduler.schedule_tasks(taskset)
                    # valid = schedule_validator(schedule, taskset)

                    import pdb
                    pdb.set_trace()

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

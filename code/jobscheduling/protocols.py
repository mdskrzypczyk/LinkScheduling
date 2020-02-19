import networkx as nx
from collections import defaultdict
from math import ceil
from copy import copy
from jobscheduling.log import LSLogger
from jobscheduling.qmath import swap_links, unswap_links, distill_links, fidelity_for_distillations, distillations_for_fidelity
from jobscheduling.task import DAGResourceSubTask, ResourceDAGTask, PeriodicResourceDAGTask
from jobscheduling.visualize import draw_DAG


logger = LSLogger()


class Protocol:
    def __init__(self, F, R, nodes):
        self.F = F
        self.R = R
        self.nodes = nodes
        self.set_duration(R)
        self.dist = 0

    def set_duration(self, R):
        if R == 0:
            self.duration = float('inf')
        else:
            self.duration = 1 / R


class LinkProtocol(Protocol):
    name_template = "LG{};{};{}"
    count = 0

    def __init__(self, F, R, nodes):
        super(LinkProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.name = self.name_template.format(self.count, *nodes)
        self.dist = 0
        LinkProtocol.count += 1

    def __copy__(self):
        return LinkProtocol(F=self.F, R=self.R, nodes=self.nodes)


class DistillationProtocol(LinkProtocol):
    name_template = "D{};{};{}"
    count = 0

    def __init__(self, F, R, protocols, nodes):
        super(DistillationProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.protocols = list(sorted(protocols, key=lambda p: p.R))
        self.durations = [protocol.duration for protocol in self.protocols]
        self.dist = max([protocol.dist for protocol in self.protocols]) + 1
        self.name = self.name_template.format(self.count, *nodes)
        DistillationProtocol.count += 1

    def set_duration(self, R):
        distillation_duration = 0.01
        self.duration = distillation_duration

    def __copy__(self):
        return DistillationProtocol(F=self.F, R=self.R, nodes=self.nodes, protocols=[copy(p) for p in self.protocols])


class SwapProtocol(Protocol):
    name_template = "S{};{}"
    count = 0

    def __init__(self, F, R, protocols, nodes):
        super(SwapProtocol, self).__init__(F=F, R=R, nodes=nodes)
        self.protocols = list(sorted(protocols, key=lambda p: p.R))
        self.durations = [protocol.duration for protocol in self.protocols]
        self.dist = max([protocol.dist for protocol in self.protocols]) + 1
        self.name = self.name_template.format(self.count, *nodes)
        SwapProtocol.count += 1

    def set_duration(self, R):
        swap_duration = 0.01
        self.duration = swap_duration

    def __copy__(self):
        return SwapProtocol(F=self.F, R=self.R, nodes=self.nodes, protocols=[copy(p) for p in self.protocols])


def create_protocol(path, G, Fmin, Rmin):
    def filter_node(node):
        return node in path

    def filter_edge(node1, node2):
        return node1 in path and node2 in path

    logger.debug("Creating protocol on path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
    subG = nx.subgraph_view(G, filter_node, filter_edge)
    try:
        protocol = esss(path, subG, Fmin, Rmin)
        if type(protocol) != Protocol and protocol is not None:
            return protocol
        else:
            return None
    except:
        logger.exception("Failed to create protocol for path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
        raise Exception()


def esss(path, G, Fmin, Rmin):
    logger.debug("ESSS on path {} with Fmin {} and Rmin {}".format(path, Fmin, Rmin))
    if len(path) == 2:
        logger.debug("Down to elementary link, finding distillation protocol")
        link_properties = G.get_edge_data(*path)['capabilities']
        protocol = get_protocol_for_link(link_properties, Fmin, Rmin, path)
        return protocol

    else:
        # Start by dividing the path in two
        lower = 1
        upper = len(path) - 1
        protocol = None
        rate = 0
        fidelity = 0
        while lower < upper:
            numL = (upper + lower + 1) // 2
            numR = len(path) + 1 - numL
            possible_protocol, Rl, Rr = find_split_path_protocol(path, G, Fmin, Rmin, numL, numR)

            if Rl == Rr:
                logger.debug("Rates on left and right paths balanced")
                return possible_protocol
            elif Rl < Rr:
                logger.debug("Rate on left path lower, extending right path")
                upper -= 1
            else:
                logger.debug("Rate on right path lower, extending left path")
                lower += 1

            if possible_protocol and possible_protocol.F >= Fmin and possible_protocol.R >= Rmin:
                if possible_protocol.R >= rate and possible_protocol.F >= fidelity:
                    protocol = possible_protocol
                    rate = protocol.R
                    fidelity = protocol.F

        return protocol


def find_split_path_protocol(path, G, Fmin, Rmin, numL, numR):
    protocols = []
    maxDistillations = 10
    for num in range(maxDistillations):
        Rlink = Rmin * (num + 1)

        # Compute minimum fidelity in order for num distillations to achieve Fmin
        Fminswap = fidelity_for_distillations(num, Fmin)
        if Fminswap == 0:
            continue
        Funswapped = unswap_links(Fminswap)

        # Search for protocols on left and right that have above properties
        protocolL = esss(path[:numL], G, Funswapped, Rlink)
        protocolR = esss(path[-numR:], G, Funswapped, Rlink)

        # Add to list of protocols
        if protocolL is not None and protocolR is not None and type(protocolL) != Protocol and type(protocolR) != Protocol:
            logger.debug("Constructing protocol")
            Fswap = swap_links(protocolL.F, protocolR.F)
            Rswap = min(protocolL.R, protocolR.R)
            swap_protocol = SwapProtocol(F=Fswap, R=Rswap, protocols=[protocolL, protocolR], nodes=[path[-numR]])
            protocol = copy(swap_protocol)
            logger.debug("Swapped link F={}".format(Fswap))
            for i in range(num):
                Fdistilled = distill_links(protocol.F, Fswap)
                Rdistilled = Rswap / (i + 2)
                protocol = DistillationProtocol(F=Fdistilled, R=Rdistilled, protocols=[protocol, copy(swap_protocol)],
                                                nodes=[path[0], path[-1]])
                logger.debug("Distilled link F={}".format(Fdistilled))

            logger.debug("Found Swap/Distill protocol achieving F={},R={},numSwappedDistills={}".format(protocol.F, protocol.R,
                                                                                                 num + 1))
            logger.debug("Underlying link protocols have Fl={},Rl={} and Fr={},Rr={}".format(protocolL.F, protocolL.R,
                                                                                      protocolR.F, protocolR.R))
            protocols.append((protocol, protocolL.R, protocolR.R))

        else:
            Rl = 0 if not protocolL else protocolL.R
            Rr = 0 if not protocolR else protocolR.R
            protocols.append((Protocol(F=0, R=0, nodes=None), Rl, Rr))

    # Choose protocol with maximum rate > Rmin
    if protocols:
        protocol, Rl, Rr = sorted(protocols, key=lambda p: p[0].R)[-1]
        logger.debug("Found Swap/Distill protocol achieving F={},R={},numSwappedDistills={}".format(protocol.F, protocol.R,
                                                                                             num + 1))
        return protocol, Rl, Rr

    else:
        return None, 0, 0


def get_protocol_for_link(link_properties, Fmin, Rmin, nodes):
    if all([R < Rmin for _, R in link_properties]):
        return None

    # Check if any single link generation protocols exist
    for F, R in link_properties:
        if R >= Rmin and F >= Fmin:
            logger.debug("Link capable of generating without distillation using F={},R={}".format(F, R))
            return LinkProtocol(F=F, R=R, nodes=nodes)

    logger.debug("Link not capable of generating without distillation")
    # Search for the link gen protocol with highest rate that satisfies fidelity
    bestR = Rmin
    bestProtocol = None
    for F, R in link_properties:
        currF = F
        currR = R
        currProtocol = LinkProtocol(F=F, R=R, nodes=nodes)
        numGens = 1
        while currR >= Rmin and currF < Fmin:
            linkProtocol = LinkProtocol(F=F, R=R, nodes=nodes)
            currF = distill_links(currF, F)
            numGens += 1
            currR = R / numGens
            currProtocol = DistillationProtocol(F=currF, R=currR, protocols=[currProtocol, linkProtocol], nodes=nodes)

        if currProtocol.F >= Fmin and currProtocol.R >= bestR:
            logger.debug("Found distillation protocol using F={},R={},numGens={}".format(currProtocol.F, currProtocol.R, numGens))
            bestR = currProtocol.R
            bestProtocol = currProtocol

    return bestProtocol


def print_protocol(protocol):
    q = [protocol]
    while q:
        p = q.pop(0)
        print(p.name, p.F, p.R)
        if type(p) == SwapProtocol or type(p) == DistillationProtocol:
            q += p.protocols


def print_resource_schedules(resource_schedules):
    schedule_length = max([rs[-1][0] + 1 for rs in resource_schedules.values() if rs])
    timeline_string = "R" + " "*max([len(r) + 1 for r in resource_schedules.keys()])
    timeline_string += ''.join(["|{:>3}".format(i) for i in range(schedule_length)])
    print(timeline_string)
    for r in sorted(resource_schedules.keys()):
        schedule_string = "{}: ".format(r)
        resource_timeline = defaultdict(lambda: None)
        for s, t in resource_schedules[r]:
            resource_timeline[s] = t
        schedule_string += ''.join(["|{:>2} ".format("V" if resource_timeline[i] is None else resource_timeline[i].name[0]) for i in range(schedule_length)])
        print(schedule_string)


def convert_protocol_to_task(request, protocol, slot_size=0.01):
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

                resources = peek_protocol_action.nodes

                dagtask = DAGResourceSubTask(name=name, c=ceil(peek_protocol_action.duration / slot_size),
                                             parents=parent_tasks[peek_protocol_action.name], dist=peek_protocol_action.dist,
                                             resources=resources)

                for parent in parent_tasks[peek_protocol_action.name]:
                    parent.add_child(dagtask)

                tasks.append(dagtask)

                if peek_child_action:
                    parent_tasks[peek_child_action.name].append(dagtask)

                last_action, _ = stack.pop()

    source, dest, fidelity, rate = request
    main_dag_task = PeriodicResourceDAGTask(name="S={}, D={}, F={}, R={}".format(source, dest, fidelity, rate), tasks=tasks,
                                            p=ceil(1 / rate / slot_size))
    return main_dag_task


def schedule_dag_for_resources(dagtask, topology):
    logger.debug("Scheduling task ASAP")
    comm_q_topology, node_topology = topology
    [sink] = dagtask.sinks
    nodes = dagtask.resources
    node_comm_resources = dict([(n, node_topology.nodes[n]["comm_qs"]) for n in nodes])
    node_storage_resources = dict([(n, node_topology.nodes[n]["storage_qs"]) for n in nodes])
    comm_to_storage_resources = dict([(comm_name, node_storage_resources[n])for n in nodes for comm_name in node_topology.nodes[n]["comm_qs"]])
    all_node_resources = dict([(n, node_topology.nodes[n]["comm_qs"] + node_topology.nodes[n]["storage_qs"]) for n in nodes])
    resource_schedules = defaultdict(list)
    stack = []
    task = sink
    last_task = None
    while stack or task:
        if task is not None:
            stack.append(task)
            task = None if not task.parents else sorted(task.parents, key=lambda pt: pt.dist)[1]

        else:
            peek_task = stack[-1]
            right_task = None if not peek_task.parents else sorted(peek_task.parents, key=lambda pt: pt.dist)[0]
            if right_task is not None and last_task != right_task:
                task = right_task

            else:
                if peek_task.name[0] == "L":
                    possible_task_resources = [node_comm_resources[n] for n in peek_task.resources]
                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules, comm_to_storage_resources)

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
                                        possible_task_resources += [[r] for r in list(sorted(p.resources))[1:4:2] if r in all_node_resources[task_node]]
                                    else:
                                        possible_task_resources += [[r] for r in p.resources if r in all_node_resources[task_node]]
                        elif pt.name[0] == "D":
                            possible_task_resources += [[r] for r in list(sorted(pt.resources))[1:4:2] if r in all_node_resources[task_node]]
                        else:
                            possible_task_resources += [[r] for r in pt.resources if r in all_node_resources[task_node]]


                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules, comm_to_storage_resources)

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
                                        possible_task_resources += [[r] for r in list(sorted(p.resources))[1:4:2] for tn in task_nodes if r in all_node_resources[tn]]
                                    else:
                                        possible_task_resources += [[r] for tn in task_nodes for sp in search_task.parents for r in sp.resources if r in all_node_resources[tn]]
                        elif pt.name[0] == "D":
                            possible_task_resources += [[r] for r in list(sorted(pt.resources))[1:4:2]]
                        else:
                            possible_task_resources += [[r] for tn in task_nodes for r in pt.resources if r in all_node_resources[tn]]

                    used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules, comm_to_storage_resources)

                last_task = stack.pop()

    sink_task = dagtask.sinks[0]
    asap_duration = sink_task.a + ceil(sink_task.c)
    asap_decoherence = get_schedule_decoherence(resource_schedules, asap_duration)
    logger.debug("Scheduling task ALAP")
    convert_task_to_alap(dagtask)
    logger.debug("ASAP Schedule latency {} total decoherence {}".format(asap_duration, asap_decoherence))
    shift_distillations_and_swaps(dagtask)
    return dagtask


def shift_distillations_and_swaps(dagtask):
    for task in sorted(dagtask.subtasks, key=lambda task: task.a):
        if task.name[0] == "D" or task.name[0] == "S":
            task.a = max([p.a + ceil(p.c) for p in task.parents])


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


def schedule_task_asap(task, task_resources, resource_schedules, storage_resources):
    earliest_start = max([0] + [p.a + ceil(p.c) for p in task.parents if set(task.resources) & set(p.resources)])
    earliest_possible_start = float('inf')
    earliest_resources = None
    earliest_mapping = None
    for resource_set in list(itertools.product(*task_resources)):
        possible_start, locked_storage_mapping = get_earliest_start_for_resources(earliest_start, task, resource_set, resource_schedules, storage_resources)

        if earliest_start <= possible_start <= earliest_possible_start and (earliest_mapping is None or len(earliest_mapping.keys()) >= len(locked_storage_mapping.keys())):
            earliest_possible_start = possible_start
            earliest_resources = resource_set
            earliest_mapping = locked_storage_mapping

        if possible_start == earliest_start:
            break

    if earliest_possible_start == float('inf') and task.name[0] == "L":
        return None

    # If the earliest_resources are locked by their last tasks, we have already confirmed there are storage resources for them
    if earliest_mapping:
        logger.debug("Changing resources due to task {}".format(task.name))
        # Iterate over the locked resources, update their last task to reference the storage resource
        for lr, sr in earliest_mapping.items():
            # Update the resource schedules with the storage resource occupying information
            lr_schedule = resource_schedules[lr]
            _, last_task = lr_schedule[-1]
            num_slots = ceil(last_task.c)
            resource_schedules[sr] += lr_schedule[-num_slots:]
            last_task.resources.remove(lr)
            last_task.resources.append(sr)
            logger.debug("Moved {} to {}".format(lr, sr))
            logger.debug("Rescheduled {} with resources {} at t={}".format(last_task.name, last_task.resources, last_task.a))

    slots = []
    for slot in range(earliest_possible_start, earliest_possible_start + ceil(task.c)):
        slots.append((slot, task))

    for r in list(set(earliest_resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    task.a = slots[0][0]
    task.resources = list(sorted(set(earliest_resources)))
    logger.debug("Scheduled {} with resources {} at t={}".format(task.name, task.resources, task.a))
    return list(set(earliest_resources))


def get_earliest_start_for_resources(earliest, task, resource_set, resource_schedules, storage_resources):
    occupied_slots = defaultdict(list)
    for r in resource_set:
        rs = resource_schedules[r]
        for slot, t in rs:
            occupied_slots[slot].append(t)

    if not occupied_slots:
        return earliest, {}

    else:
        occupied_ranges = list(to_ranges(occupied_slots.keys()))
        _, e = occupied_ranges[-1]
        last_tasks = []
        for r in resource_set:
            rs = resource_schedules[r]
            filtered_schedule = list(sorted(filter(lambda slot_info: slot_info[0] <= e, rs)))
            if filtered_schedule:
                last_task_slot, last_task = filtered_schedule[-1]
                last_tasks.append((r, last_task))

        if not (task.name[0] == "L" and any([task_locks_resource(t, [r]) for r, t in last_tasks])):
            if e >= earliest:
                return e + 1, {}
            else:
                return earliest, {}

        elif task.name[0] == "L" and any([task_locks_resource(t, [r]) for r, t in last_tasks]):
            locked_resources = set([r for r, t in last_tasks if task_locks_resource(t, [r])])
            lr_to_sr = defaultdict(lambda: None)
            sr_to_lr = defaultdict(lambda: None)
            for locked_r in locked_resources:
                # Find a storage resource to move the locked resource into (check whether the storage resources are also locked)
                for storage_r in storage_resources[locked_r]:
                    if lr_to_sr[locked_r]:
                        continue
                    rs = resource_schedules[storage_r]
                    filtered_schedule = list(sorted(filter(lambda slot_info: slot_info[0] <= e, rs)))

                    if filtered_schedule:
                        last_task_slot, last_task = filtered_schedule[-1]
                        if not task_locks_resource(last_task, [storage_r]) and sr_to_lr[storage_r] is None:
                            lr_to_sr[locked_r] = storage_r
                            sr_to_lr[storage_r] = locked_r

                    elif sr_to_lr[storage_r] is None:
                        lr_to_sr[locked_r] = storage_r
                        sr_to_lr[storage_r] = locked_r

                # If non available, return float('inf')
                if lr_to_sr[locked_r] is None:
                    return float('inf'), {}

            # Return the slot where the new action can start
            if e >= earliest:
                return e + 1, lr_to_sr
            else:
                return earliest, lr_to_sr
        else:
            return float('inf'), {}


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
    logger.debug("ALAP Schedule latency {} total decoherence {}".format(alap_latency, alap_decoherence))


def schedule_task_alap(task, resource_schedules):

    possible = [task.a]
    child_starts = [ct.a - ceil(task.c) for ct in task.children if set(task.resources) & set(ct.resources)]
    if child_starts:
        possible += [min(child_starts)]
    latest = max(possible)

    latest = min([latest] + [resource_schedules[r][0][0] - ceil(task.c) for r in task.resources if resource_schedules[r]])

    slots = [(s, task) for s in range(latest, latest + ceil(task.c))]
    for r in list(set(task.resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    if slots[0][0] != task.a:
        logger.debug("Moved task {} from t={} to t={}".format(task.name, task.a, slots[0][0]))
    task.a = slots[0][0]
    logger.debug("Scheduled {} with resources {} at t={}".format(task.name, task.resources, task.a))


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



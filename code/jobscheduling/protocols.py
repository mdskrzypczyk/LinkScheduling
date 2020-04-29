import itertools
from collections import defaultdict
from math import ceil
from jobscheduling.log import LSLogger
from jobscheduling.task import DAGResourceSubTask, PeriodicBudgetResourceDAGTask, get_dag_exec_time
from intervaltree import Interval, IntervalTree
from random import randint

logger = LSLogger()


def print_resource_schedules(resource_schedules):
    schedule_length = max([rs[-1][0] + 1 for rs in resource_schedules.values() if rs])
    timeline_string = " R" + " " * max([len(r) + 1 for r in resource_schedules.keys()])
    timeline_string += ''.join(["|{:>3} ".format(i) for i in range(schedule_length)])
    print(timeline_string)
    for r in sorted(resource_schedules.keys()):
        schedule_string = "{:>5}: ".format(r)
        resource_timeline = defaultdict(lambda: None)
        for s, t in resource_schedules[r]:
            resource_timeline[s] = t
        schedule_string += ''.join(["|{:>3} ".format("V" if resource_timeline[i] is None else
                                                     resource_timeline[i].name[0]) for i in range(schedule_length)])
        print(schedule_string)


def get_protocol_rate(demand, protocol, topology):
    slot_size = 0.01
    task = convert_protocol_to_task(demand, protocol, slot_size)
    scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, topology)
    latency = scheduled_task.c * slot_size
    achieved_rate = 1 / latency
    return achieved_rate


def convert_protocol_to_task(request, protocol, slot_size=0.1):
    tasks = []
    labels = {
        "L": 0,
        "D": 0,
        "S": 0
    }

    stack = []
    protocol_action, child_action = (protocol, None)
    parent_tasks = defaultdict(list)
    last_action = None
    while stack or protocol_action is not None:
        if protocol_action is not None:
            stack.append((protocol_action, child_action))

            if protocol_action.name[0] != "L":
                left_protocol_action = protocol_action.protocols[0]
                child_action = protocol_action
                protocol_action = left_protocol_action

            else:
                protocol_action = None
                child_action = None

        else:
            peek_protocol_action, peek_child_action = stack[-1]
            right_protocol_action = None if peek_protocol_action.name[0] == "L" else peek_protocol_action.protocols[1]
            if right_protocol_action is not None and last_action != right_protocol_action:
                protocol_action = right_protocol_action
                child_action = peek_protocol_action

            else:
                name = peek_protocol_action.name
                suffix = name.split(';')[1:]
                name = ';'.join([peek_protocol_action.name[0]] + [str(labels[peek_protocol_action.name[0]])] + suffix)
                labels[peek_protocol_action.name[0]] += 1

                resources = peek_protocol_action.nodes

                dagtask = DAGResourceSubTask(name=name, c=ceil(peek_protocol_action.duration / slot_size),
                                             parents=parent_tasks[peek_protocol_action.name],
                                             dist=peek_protocol_action.dist, resources=resources)

                F = round(peek_protocol_action.F, 2)
                R = round(peek_protocol_action.R, 2)
                dagtask.description = "F={}, R={}".format(F, R)

                for parent in parent_tasks[peek_protocol_action.name]:
                    parent.add_child(dagtask)

                tasks.append(dagtask)

                if peek_child_action:
                    parent_tasks[peek_child_action.name].append(dagtask)

                last_action, _ = stack.pop()

    source, dest, fidelity, rate = request
    task_id = randint(0, 100)
    task_name = "S={}, D={}, F={}, R={}, ID={}".format(source, dest, fidelity, rate, task_id)
    main_dag_task = PeriodicBudgetResourceDAGTask(name=task_name, tasks=tasks, p=ceil(1 / rate / slot_size), k=100)
    return main_dag_task


def schedule_dag_for_resources(dagtask, topology):
    logger.debug("Scheduling task ASAP")
    asap_latency, asap_decoherence, asap_correct = schedule_dag_asap(dagtask, topology)
    logger.debug("ASAP Schedule latency {} total decoherence {}, correct={}".format(asap_latency, asap_decoherence,
                                                                                    asap_correct))

    logger.debug("Scheduling task ALAP")
    alap_latency, alap_decoherence, alap_correct = convert_task_to_alap(dagtask)
    logger.debug("ALAP Schedule latency {} total decoherence {}, correct={}".format(alap_latency, alap_decoherence,
                                                                                    alap_correct))

    logger.debug("Shifting SWAPs and Distillations")
    shift_latency, shift_decoherence, shift_correct = shift_distillations_and_swaps(dagtask)
    logger.debug("Shifted Schedule latency {} total decoherence {}, correct={}".format(shift_latency, shift_decoherence,
                                                                                       shift_correct))

    decoherences = (asap_decoherence, alap_decoherence, shift_decoherence)
    correct = (asap_correct and alap_correct and shift_correct)
    dagtask.c = get_dag_exec_time(dagtask)
    return dagtask, decoherences, correct


def schedule_dag_asap(dagtask, topology):
    comm_q_topology, node_topology = topology
    [sink] = dagtask.sinks
    nodes = dagtask.resources
    node_comm_resources = dict([(n, node_topology.nodes[n]["comm_qs"]) for n in nodes])
    node_storage_resources = dict([(n, node_topology.nodes[n]["storage_qs"]) for n in nodes])
    comm_to_storage_resources = dict(
        [(comm_name, node_storage_resources[n]) for n in nodes for comm_name in node_topology.nodes[n]["comm_qs"]])
    all_node_resources = dict(
        [(n, node_topology.nodes[n]["comm_qs"] + node_topology.nodes[n]["storage_qs"]) for n in nodes])
    resource_schedules = defaultdict(list)
    stack = []
    task = sink
    last_task = None
    for task in dagtask.subtasks:
        task.parents = list(sorted(task.parents, key=lambda pt: (pt.dist, pt.name)))

    while stack or task:
        if task is not None:
            stack.append(task)
            task = None if not task.parents else task.parents[1]

        else:
            peek_task = stack[-1]
            right_task = None if not peek_task.parents else peek_task.parents[0]
            if right_task is not None and last_task != right_task:
                task = right_task

            else:
                possible_task_resources = get_possible_resources_for_task(peek_task, node_comm_resources,
                                                                          all_node_resources)
                schedule_task_asap(peek_task, possible_task_resources, resource_schedules, comm_to_storage_resources)

                last_task = stack.pop()

    dagtask.resources = list(set([r for subtask in dagtask.subtasks for r in subtask.resources]))
    asap_latency = get_dag_exec_time(dagtask)
    resource_schedules = defaultdict(list)
    for task in dagtask.subtasks:
        slots = [(task.a + i, task) for i in range(ceil(task.c))]
        for resource in task.resources:
            resource_schedules[resource] += slots

    for resource in resource_schedules.keys():
        resource_schedules[resource] = list(sorted(resource_schedules[resource]))

    asap_decoherence = get_schedule_decoherence(dagtask.get_resource_schedules(), asap_latency)
    asap_correct = verify_dag(dagtask)
    return asap_latency, asap_decoherence, asap_correct


def get_possible_resources_for_task(task, node_comm_resources, all_node_resources):
    if task.name[0] == "L":
        possible_task_resources = [node_comm_resources[n] for n in task.resources]
        if len(possible_task_resources) not in [2, 3, 4]:
            import pdb
            pdb.set_trace()

    else:
        possible_task_resources = get_resources_from_parents(task, all_node_resources)
        if task.name[0] == "D" and len(possible_task_resources) != 4 or task.name[0] == "S" and \
                len(possible_task_resources) != 2:
            import pdb
            pdb.set_trace()

    return possible_task_resources


def get_resources_from_parents(task, all_node_resources):
    possible_task_resources = []
    task_nodes = task.resources

    for pt in task.parents:
        if pt.name[0] == "S":
            q = [pt]
            while q:
                search_task = q.pop(0)
                if search_task.name[0] == "S":
                    for p in search_task.parents:
                        q.append(p)
                else:
                    possible_task_resources += get_resources_for_task_nodes(search_task, task_nodes, all_node_resources)
        else:
            possible_task_resources += get_resources_for_task_nodes(pt, task_nodes, all_node_resources)

    return possible_task_resources


def get_resources_for_task_nodes(task, task_nodes, all_node_resources):
    if task.name[0] == "D":
        return [[r] for r in list(sorted(task.locked_resources)) for tn in task_nodes if r in all_node_resources[tn]]
    elif task.name[0] == "L":
        return [[r] for tn in task_nodes for r in task.locked_resources if r in all_node_resources[tn]]
    else:
        import pdb
        pdb.set_trace()


def schedule_task_asap(task, task_resources, resource_schedules, storage_resources):
    earliest_possible_start = float('inf')
    earliest_resources = None
    earliest_mapping = None

    for resource_set in list(itertools.product(*task_resources)):
        earliest_start = max([0] + [p.a + ceil(p.c) for p in task.parents if set(resource_set) & set(p.resources)])

        possible_start, locked_storage_mapping = get_earliest_start_for_resources(earliest_start, task, resource_set,
                                                                                  resource_schedules, storage_resources)
        if earliest_start <= possible_start < earliest_possible_start and \
                (earliest_mapping is None or len(earliest_mapping.keys()) >= len(locked_storage_mapping.keys())):
            earliest_possible_start = possible_start
            earliest_resources = resource_set
            earliest_mapping = locked_storage_mapping

        if possible_start == earliest_start:
            break

    if earliest_possible_start == float('inf') and task.name[0] == "L":
        logger.debug("Failed to schedule {}".format(task.name))
        return None

    earliest_resources = list(set(earliest_resources))
    slots = []
    for slot in range(earliest_possible_start, earliest_possible_start + ceil(task.c)):
        slots.append((slot, task))

    sr_mapping = defaultdict(lambda: (float('inf'), None))
    if task.name[0] == "L" and earliest_mapping == {}:
        for lr in earliest_resources:
            rs = resource_schedules[lr]
            if rs:
                last_slot, last_task = rs[-1]
                if task_locks_resource(last_task, [lr]):
                    continue
            for sr in storage_resources[lr]:
                rs = resource_schedules[sr]
                if rs:
                    last_slot, last_task = rs[-1]
                    earliest_slot, _ = sr_mapping[lr]
                    if not task_locks_resource(last_task, [sr]) and last_slot + 1 < earliest_slot:
                        sr_mapping[lr] = (last_slot, sr)

                elif -1 < sr_mapping[lr][0]:
                    sr_mapping[lr] = (-1, sr)

            rs = resource_schedules[lr]
            if rs:
                last_slot, last_task = rs[-1]
            else:
                last_slot = 0

            if sr_mapping[lr][1] is None:
                sr_mapping[lr] = (last_slot, lr)

        for t, sr in sr_mapping.values():
            if sr is not None and t + 1 <= earliest_possible_start:
                earliest_resources.append(sr)

    for r in list(set(earliest_resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    task.a = earliest_possible_start
    task.resources = list(sorted(set(earliest_resources)))
    if task.name[0] == "L" and earliest_mapping == {}:
        task.locked_resources = list(task.resources)
        for lr in sr_mapping.keys():
            if sr_mapping[lr][1] != lr and sr_mapping[lr][1] in earliest_resources:
                task.locked_resources.remove(lr)

    elif task.name[0] == "D":
        task.locked_resources = task.resources[1:4:2]

    logger.debug("Scheduled {} with resources {} and locked resources {} at t={}".format(task.name, task.resources,
                                                                                         task.locked_resources, task.a))
    return list(set(earliest_resources))


def get_earliest_start_for_resources(earliest, task, resource_set, resource_schedules, storage_resources):
    # Always get the first earliest pair of communication qubits and storage qubits, if no storage qubits left leave
    # the link in the comm qubits
    occupied_slots = defaultdict(list)

    for r in resource_set:
        rs = resource_schedules[r]
        for slot, t in rs:
            occupied_slots[slot].append(t)

    if not occupied_slots:
        return earliest, {}

    else:
        occupied_ranges = list(sorted(to_ranges(occupied_slots.keys())))
        _, e = occupied_ranges[-1]

        # Find an immediate mapping to storage resources
        if task.name[0] == "L":
            sr_mapping = defaultdict(lambda: (float('inf'), None))
            for lr in resource_set:
                rs = resource_schedules[lr]
                if rs:
                    last_slot, last_task = rs[-1]
                    if task_locks_resource(last_task, [lr]):
                        return float('inf'), {}

                for sr in storage_resources[lr]:
                    rs = resource_schedules[sr]
                    if rs:
                        last_slot, last_task = rs[-1]
                        earliest_slot, _ = sr_mapping[lr]
                        if not task_locks_resource(last_task, [sr]) and last_slot + 1 < earliest_slot:
                            sr_mapping[lr] = (last_slot, sr)

                    elif -1 < sr_mapping[lr][0]:
                        sr_mapping[lr] = (-1, sr)

                rs = resource_schedules[lr]
                if rs:
                    last_slot, last_task = rs[-1]
                else:
                    last_slot = 0

                if sr_mapping[lr][1] is None:
                    sr_mapping[lr] = (last_slot, lr)

            if all([sr is not None for _, sr in sr_mapping.values()]):
                earliest_resource_time = max([e] + [t for t, _ in sr_mapping.values()])
                return earliest_resource_time + 1, {}

        last_tasks = []
        for r in resource_set:
            rs = resource_schedules[r]
            if rs:
                last_task_slot, last_task = rs[-1]
                last_tasks.append((r, last_task))

        if not (task.name[0] == "L" and any([task_locks_resource(t, [r]) for r, t in last_tasks])):
            if e >= earliest:
                return e + 1, {}
            else:
                return earliest, {}
        else:
            return float('inf'), {}


def to_ranges(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def task_locks_resource(task, resources):
    link_lock = (task.name[0] == "L" and any([r in task.locked_resources for r in resources]))
    distill_lock = (task.name[0] == "D" and any([r in list(sorted(task.locked_resources)) for r in resources]))
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

    alap_latency = get_dag_exec_time(dagtask)
    alap_decoherence = get_schedule_decoherence(dagtask.get_resource_schedules(), alap_latency)
    alap_correct = verify_dag(dagtask)
    return alap_latency, alap_decoherence, alap_correct


def schedule_task_alap(task, resource_schedules):
    possible = [task.a]
    child_starts = [ct.a - ceil(task.c) for ct in task.children if set(task.resources) & set(ct.resources)]
    if child_starts:
        possible += [min(child_starts)]
    latest = max(possible)

    latest = min([latest] + [resource_schedules[r][0][0] - ceil(task.c) for r in task.resources
                             if resource_schedules[r]])

    slots = [(s, task) for s in range(latest, latest + ceil(task.c))]
    for r in list(set(task.resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    if latest != task.a:
        logger.debug("Moved task {} from t={} to t={}".format(task.name, task.a, latest))
    task.a = latest
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


def shift_distillations_and_swaps(dagtask):
    resource_schedules = defaultdict(IntervalTree)
    for link_task in dagtask.sources:
        begin = link_task.a
        end = begin + ceil(link_task.c)
        interval = Interval(begin, end, link_task)
        for resource in link_task.resources:
            resource_schedules[resource].add(interval)

    for task in sorted(dagtask.subtasks, key=lambda task: task.a):
        if task.name[0] == "D" or task.name[0] == "S":
            parent_ends = [p.a + ceil(p.c) for p in task.parents if set(p.resources) & set(task.resources)]
            parent_task_end = max(parent_ends) if parent_ends else 0
            resource_availabilities = []
            for resource in task.resources:
                interval_set = resource_schedules[resource].envelop(0, task.a)
                if interval_set:
                    interval = sorted(interval_set)[-1]
                    resource_availabilities.append(interval.end)

            earliest_start = max([parent_task_end] + resource_availabilities)
            task.a = earliest_start
            if task.c > 0:
                interval = Interval(begin=task.a, end=task.a + ceil(task.c), data=task)
                for resource in task.resources:
                    resource_schedules[resource].add(interval)

    resource_schedules_new = {}
    for resource in resource_schedules.keys():
        resource_schedule = []
        for interval in resource_schedules[resource]:
            task = interval.data
            slots = [(task.a + i, task) for i in range(ceil(task.c))]
            resource_schedule += slots
        resource_schedules_new[resource] = list(sorted(resource_schedule))

    shift_latency = get_dag_exec_time(dagtask)
    shift_decoherence = get_schedule_decoherence(dagtask.get_resource_schedules(), shift_latency)
    shift_correct = verify_dag(dagtask)

    return shift_latency, shift_decoherence, shift_correct


def verify_dag(dagtask, node_resources=None):
    resource_intervals = defaultdict(IntervalTree)
    valid = True
    for subtask in dagtask.subtasks:
        for child in subtask.children:
            if child.a < subtask.a + ceil(subtask.c) and set(child.resources) & set(subtask.resources):
                valid = False

        if subtask.name[0] == "L" and \
                (len(set(subtask.locked_resources)) != 2 or len(set(subtask.resources)) not in [2, 3, 4]):
            logger.error("Link generation subtask has incorrect set of resources")
            valid = False

        if subtask.name[0] == "S" and len(set(subtask.resources)) != 2:
            logger.error("Swapping subtask has incorrect set of resources")
            import pdb
            pdb.set_trace()
            valid = False

        if subtask.name[0] == "D" and (len(set(subtask.resources)) != 4 or len(set(subtask.locked_resources)) != 2):
            logger.error("Distillation subtask has incorrect set of resources")
            import pdb
            pdb.set_trace()
            valid = False

        if subtask.c > 0:
            subtask_interval = Interval(subtask.a, subtask.a + subtask.c, subtask)
            for resource in subtask.resources:
                if node_resources and (resource not in node_resources):
                    continue

                if resource_intervals[resource].overlap(subtask_interval.begin, subtask_interval.end):
                    overlapping = sorted(resource_intervals[resource][subtask_interval.begin:subtask_interval.end])[0]
                    logger.error("Subtask {} overlaps at resource {}"
                                 " during interval {},{} with task {}".format(subtask.name, resource, overlapping.begin,
                                                                              overlapping.end, overlapping.data.name))
                    valid = False
                resource_intervals[resource].add(subtask_interval)

    for resource in resource_intervals.keys():
        sorted_intervals = sorted(resource_intervals[resource])
        for iv1, iv2 in zip(sorted_intervals, sorted_intervals[1:]):
            t1 = iv1.data
            t2 = iv2.data
            if t1.name[0] == "L" and t2.name == "L":
                logger.error("Consecutive link generation subtasks on common resource")
                valid = False

            elif t1.name[0] == "D" and t2.name == "L" and any([r in t1.locked_resources for r in t2.locked_resources]):
                logger.error("Distillation followed by link generation on same resource")
                valid = False

    return valid


def get_schedule_decoherence(resource_schedules, completion_time):
    total_decoherence = 0
    resource_decoherences = {}
    for r in sorted(resource_schedules.keys()):
        resource_decoherence = 0
        rs = resource_schedules[r]
        for slot, task in rs:
            if task.name[0] == "O":
                resource_decoherence += 1

        resource_decoherences[r] = resource_decoherence
        total_decoherence += resource_decoherence

    return total_decoherence

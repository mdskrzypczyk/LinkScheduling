import itertools
from collections import defaultdict
from math import ceil
from jobscheduling.log import LSLogger
from jobscheduling.protocolgen import LinkProtocol, DistillationProtocol, SwapProtocol
from jobscheduling.task import DAGResourceSubTask, ResourceDAGTask, PeriodicResourceDAGTask
from jobscheduling.visualize import draw_DAG
from intervaltree import Interval, IntervalTree


logger = LSLogger()


def print_resource_schedules(resource_schedules):
    schedule_length = max([rs[-1][0] + 1 for rs in resource_schedules.values() if rs])
    timeline_string = "R" + " "*max([len(r) + 1 for r in resource_schedules.keys()])
    timeline_string += ''.join(["|{:>3} ".format(i) for i in range(schedule_length)])
    print(timeline_string)
    for r in sorted(resource_schedules.keys()):
        schedule_string = "{:>5}: ".format(r)
        resource_timeline = defaultdict(lambda: None)
        for s, t in resource_schedules[r]:
            resource_timeline[s] = t
        schedule_string += ''.join(["|{:>3} ".format("V" if resource_timeline[i] is None else resource_timeline[i].name[0]) for i in range(schedule_length)])
        print(schedule_string)


def convert_protocol_to_task(request, protocol, slot_size=0.1):
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
                used_resources = schedule_task_asap(peek_task, possible_task_resources, resource_schedules,
                                                    comm_to_storage_resources)

                last_task = stack.pop()

    sink_task = dagtask.sinks[0]
    asap_latency = sink_task.a + ceil(sink_task.c)
    resource_schedules = defaultdict(list)
    for task in dagtask.subtasks:
        slots = [(task.a + i, task) for i in range(ceil(task.c))]
        for resource in task.resources:
            resource_schedules[resource] += slots

    for resource in resource_schedules.keys():
        resource_schedules[resource] = list(sorted(resource_schedules[resource]))

    asap_decoherence = get_schedule_decoherence(resource_schedules, asap_latency)
    asap_correct = verify_dag(dagtask)
    return asap_latency, asap_decoherence, asap_correct


def get_possible_resources_for_task(task, node_comm_resources, all_node_resources):
    if task.name[0] == "L":
        possible_task_resources = [node_comm_resources[n] for n in task.resources]

    else:
        possible_task_resources = get_resources_from_parents(task, all_node_resources)

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
        return [[r] for r in list(sorted(task.resources))[1:4:2] for tn in task_nodes if r in all_node_resources[tn]]
    elif task.name[0] == "L":
        return [[r] for tn in task_nodes for r in task.resources if r in all_node_resources[tn]]
    else:
        import pdb
        pdb.set_trace()


def schedule_task_asap(task, task_resources, resource_schedules, storage_resources):
    earliest_possible_start = float('inf')
    earliest_resources = None
    earliest_mapping = None

    for resource_set in list(itertools.product(*task_resources)):
        earliest_start = max([0] + [p.a + ceil(p.c) for p in task.parents if set(resource_set) & set(p.resources)])

        possible_start, locked_storage_mapping = get_earliest_start_for_resources(earliest_start, task, resource_set, resource_schedules, storage_resources)
        if earliest_start <= possible_start < earliest_possible_start and (earliest_mapping is None or len(earliest_mapping.keys()) >= len(locked_storage_mapping.keys())):
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
        task_starts = defaultdict(int)
        for lr in earliest_mapping.keys():
            lr_schedule = resource_schedules[lr]
            _, last_task = lr_schedule[-1]
            reassigned_start, sr = earliest_mapping[lr]
            task_starts[last_task.name] = max(task_starts[last_task.name], reassigned_start)

        for lr in sorted(earliest_mapping.keys()):
            # Update the resource schedules with the storage resource occupying information
            _, sr = earliest_mapping[lr]
            lr_schedule = resource_schedules[lr]
            _, last_task = lr_schedule[-1]
            reassigned_start = task_starts[last_task.name]
            num_slots = ceil(last_task.c)
            slots_to_modify = lr_schedule[-num_slots:]
            resource_schedules[lr] = lr_schedule[:-num_slots]
            for i in range(num_slots):
                prev_slot = slots_to_modify[i]
                slot_num, _ = prev_slot
                new_slot = (reassigned_start + i, last_task)
                resource_schedules[lr].append(new_slot)
                resource_schedules[sr].append(new_slot)

            last_task.a = reassigned_start
            last_task.resources.remove(lr)
            last_task.resources = list(sorted(last_task.resources + [sr]))
            logger.debug("Moved {} to {}".format(lr, sr))
            logger.debug("Rescheduled {} with resources {} at t={}".format(last_task.name, last_task.resources, last_task.a))

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
            t = max([s[0] for s in sr_mapping.values()])
            if sr is not None and t + 1 <= earliest_possible_start:
                earliest_resources.append(sr)

    for r in list(set(earliest_resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    if task.name[0] == "L" and earliest_mapping == {}:
        for lr in sr_mapping.keys():
            t = max([s[0] for s in sr_mapping.values()])
            if t + 1 <= earliest_possible_start:
                earliest_resources.remove(lr)

    task.a = slots[0][0]
    task.resources = list(sorted(set(earliest_resources)))
    logger.debug("Scheduled {} with resources {} at t={}".format(task.name, task.resources, task.a))
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

            if all([sr is not None for _, sr in sr_mapping.values()]):
                earliest_resource_time = max([e] + [t for t, _ in sr_mapping.values()])
                return earliest_resource_time + 1, {}

        last_tasks = []
        for r in resource_set:
            rs = resource_schedules[r]
            # filtered_schedule = list(sorted(filter(lambda slot_info: slot_info[0] <= e, rs)))
            if rs:
                last_task_slot, last_task = rs[-1]
                last_tasks.append((r, last_task))

        if not (task.name[0] == "L" and any([task_locks_resource(t, [r]) for r, t in last_tasks])):
            if e >= earliest:
                return e + 1, {}
            else:
                return earliest, {}

        elif task.name[0] == "L" and any([task_locks_resource(t, [r]) for r, t in last_tasks]):
            all_locked_resources = set([(r, t) for _, t in last_tasks for r in t.resources if r in storage_resources.keys() and task_locks_resource(t, [r])])
            must_map_resources = set([(r, t) for r, t in last_tasks if task_locks_resource(t, [r])])
            optional_map_resources = set([(r, t) for r, t in all_locked_resources if resource_schedules[r][-1][1] == t]) - must_map_resources

            mapping = defaultdict(dict)
            lr_to_sr = defaultdict(lambda: None)
            sr_to_lr = defaultdict(lambda: None)
            remapped_locked_resources = set()
            for locked_r, locking_task in sorted(must_map_resources):
                earliest_storage_time = float('inf')
                earliest_storage_resource = None
                for storage_r in storage_resources[locked_r]:
                    if storage_r in sr_to_lr.keys():
                        continue

                    rs = resource_schedules[storage_r]

                    if rs:
                        last_task_slot, last_task = rs[-1]
                        if (not task_locks_resource(last_task, [storage_r])) and (sr_to_lr[storage_r] is None):
                            if last_task_slot + ceil(locking_task.c) < earliest_storage_time:
                                storage_time = max(last_task_slot + 1, locking_task.a)
                                if storage_time < earliest_storage_time:
                                    earliest_storage_time = storage_time
                                    earliest_storage_resource = storage_r

                    elif locking_task.a < earliest_storage_time:
                        earliest_storage_resource = storage_r
                        earliest_storage_time = locking_task.a

                # If none available, return float('inf')
                if earliest_storage_resource is None and (locked_r, locking_task) in must_map_resources:
                    return float('inf'), {}

                elif earliest_storage_resource:
                    lr_to_sr[locked_r] = (earliest_storage_time, earliest_storage_resource)
                    mapping[locking_task.name][locked_r] = (earliest_storage_time, earliest_storage_resource)
                    sr_to_lr[earliest_storage_resource] = locked_r
                    remapped_locked_resources |= {(locked_r, locking_task)}

            for locked_r, locking_task in sorted(optional_map_resources):
                if locked_r in lr_to_sr.keys():
                    continue

                earliest_storage_time = float('inf')
                earliest_storage_resource = None
                for storage_r in storage_resources[locked_r]:
                    if storage_r in sr_to_lr.keys():
                        continue

                    rs = resource_schedules[storage_r]

                    if rs:
                        last_task_slot, last_task = rs[-1]
                        if (not task_locks_resource(last_task, [storage_r])) and (sr_to_lr[storage_r] is None):
                            if last_task_slot + ceil(locking_task.c) < earliest_storage_time:
                                storage_time = max(last_task_slot + 1, locking_task.a)
                                if storage_time < earliest_storage_time:
                                    earliest_storage_time = storage_time
                                    earliest_storage_resource = storage_r

                    elif locking_task.a < earliest_storage_time:
                        earliest_storage_resource = storage_r
                        earliest_storage_time = locking_task.a

                # If none available, return float('inf')
                if earliest_storage_resource:
                    lr_to_sr[locked_r] = (earliest_storage_time, earliest_storage_resource)
                    mapping[locking_task.name][locked_r] = (earliest_storage_time, earliest_storage_resource)
                    sr_to_lr[earliest_storage_resource] = locked_r
                    remapped_locked_resources |= {(locked_r, locking_task)}

            new_task_start = max([e] + [lr_to_sr[lr][0] + ceil(lt.c) - 1 for lr, lt in remapped_locked_resources])

            # Return the slot where the new action can start
            if new_task_start >= earliest:
                return new_task_start + 1, lr_to_sr
            else:
                return earliest, lr_to_sr
        else:
            return float('inf'), {}


def to_ranges(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


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
    alap_correct = verify_dag(dagtask)
    return alap_latency, alap_decoherence, alap_correct


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
            parent_task_end = max(parent_ends) if parent_ends else task.a
            resource_availabilities = []
            for resource in task.resources:
                interval_set = resource_schedules[resource].envelop(0, task.a)
                if interval_set:
                    interval = sorted(interval_set)[-1]
                    resource_availabilities.append(interval.end)

            earliest_start = max([parent_task_end] + resource_availabilities)
            task.a = earliest_start
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

    sink_task = dagtask.sinks[0]
    shift_latency = sink_task.a + ceil(sink_task.c)
    shift_decoherence = get_schedule_decoherence(resource_schedules_new, shift_latency)
    shift_correct = verify_dag(dagtask)

    return shift_latency, shift_decoherence, shift_correct


def verify_dag(dagtask, node_resources=None):
    resource_intervals = defaultdict(IntervalTree)
    valid = True
    for subtask in dagtask.subtasks:
        for child in subtask.children:
            if child.a < subtask.a + ceil(subtask.c) and set(child.resources) & set(subtask.resources):
                valid = False

        if subtask.name[0] == "L" and len(set(subtask.resources)) != 2:
            import pdb
            pdb.set_trace()

        if subtask.name[0] == "S" and len(set(subtask.resources)) != 2:
            import pdb
            pdb.set_trace()

        if subtask.name[0] == "D" and len(set(subtask.resources)) != 4:
            import pdb
            pdb.set_trace()


        subtask_interval = Interval(subtask.a, subtask.a + subtask.c, subtask)
        for resource in subtask.resources:
            if node_resources and (resource not in node_resources):
                continue

            if resource_intervals[resource].overlap(subtask_interval.begin, subtask_interval.end):
                overlapping = sorted(resource_intervals[resource][subtask_interval.begin:subtask_interval.end])[0]
                print("Subtask {} overlaps at resource {} during interval {},{} with task {}".format(subtask.name, resource, overlapping.begin, overlapping.end, overlapping.data.name))
                valid = False
            resource_intervals[resource].add(subtask_interval)

    for resource in resource_intervals.keys():
        sorted_intervals = sorted(resource_intervals[resource])
        for iv1, iv2 in zip(sorted_intervals, sorted_intervals[1:]):
            t1 = iv1.data
            t2 = iv2.data
            if t1.name[0] == "L" and t2.name == "L":
                valid = False

            elif t1.name[0] == "D" and t2.name == "L" and any([r in t1.resources[1:4:2] for r in t2.resources]):
                valid = False

    return valid


def get_schedule_decoherence(resource_schedules, completion_time):
    total_decoherence = 0
    resource_decoherences = {}
    for r in sorted(resource_schedules.keys()):
        resource_decoherence = 0
        rs = resource_schedules[r]
        for (s1, t1), (s2, t2) in zip(rs, rs[1:]):
            if (t1.name[0] == "L" or (t1.name[0] == "D" and r in t1.resources[1:4:2])) and t2 != t1 and s1 + ceil(t1.c) != s2:
                resource_decoherence += (s2 - 1 - s1)

        if rs:
            s, t = rs[-1]
            if t.name[0] == "L" or (t.name[0] == "D" and r in t.resources[1:4:2]):
                resource_decoherence += (completion_time - 1 - s)

        resource_decoherences[r] = resource_decoherence
        total_decoherence += resource_decoherence

    return total_decoherence

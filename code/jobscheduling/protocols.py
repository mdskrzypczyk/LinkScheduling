import itertools
from collections import defaultdict
from math import ceil
from jobscheduling.log import LSLogger
from jobscheduling.task import DAGResourceSubTask, PeriodicBudgetResourceDAGTask, get_dag_exec_time
from intervaltree import Interval, IntervalTree
from random import randint

logger = LSLogger()


def get_protocol_rate(demand, protocol, topology, slot_size=0.01):
    """
    Obtains the rate that a protocol can be executed at
    :param demand: type tuple
        Tuple describing the source, destination, Fmin, and Rmin
    :param protocol: type Protocol
        The repeater protocol to obtain the rate of
    :param topology: type tuple
        Tuple of the communication resource and connectivity graphs
    :param slot_size: type float
        The size of a slot in seconds
    :return: type float
        The maximum rate the protocol may be executed
    """
    # Convert the protocol into a task
    task = convert_protocol_to_task(demand, protocol, slot_size)

    # Schedule the task onto qubits (makes the concrete repeater protocol)
    scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, topology)

    # Get the rate
    latency = scheduled_task.c * slot_size
    achieved_rate = 1 / latency
    return achieved_rate


def convert_protocol_to_task(request, protocol, slot_size=0.1):
    """
    Converts a protocol to a DAGTask that is not mapped to time or hardware
    :param request: type tuple
        Tuple describing the source, destination, Fmin, Rmin of network demand
    :param protocol: type Protocol
        The protocol to be converted to a DAGTask
    :param slot_size: type float
        Size of a time slot in seconds
    :return: type DAGTask
        DAGTask representing the repeater protocol
    """
    # Track individual tasks
    tasks = []

    # Task labels for type of operation
    labels = {
        "L": 0,
        "D": 0,
        "S": 0
    }

    # Stack for working through tasks
    stack = []

    # Track the current protocol action
    protocol_action, child_action = (protocol, None)

    # Track the parent/child relationship in protocol
    parent_tasks = defaultdict(list)
    last_action = None

    # Create the DAGTask for the protocol by doing an post-order traversal of protocol operations
    while stack or protocol_action is not None:
        # Add the current protocol operation to the stack
        if protocol_action is not None:
            stack.append((protocol_action, child_action))

            # If the left child is swap or distill then move to it
            if protocol_action.name[0] != "L":
                left_protocol_action = protocol_action.protocols[0]
                child_action = protocol_action
                protocol_action = left_protocol_action

            else:
                protocol_action = None
                child_action = None

        # Pop the last added protocol action
        else:
            peek_protocol_action, peek_child_action = stack[-1]

            # Move to the right child if it is a swap or distill
            right_protocol_action = None if peek_protocol_action.name[0] == "L" else peek_protocol_action.protocols[1]
            if right_protocol_action is not None and last_action != right_protocol_action:
                protocol_action = right_protocol_action
                child_action = peek_protocol_action

            # Otherwise we label the action and add it to the DAGTask
            else:
                # Create the SubTask for this operation
                name = peek_protocol_action.name
                suffix = name.split(';')[1:]
                name = ';'.join([peek_protocol_action.name[0]] + [str(labels[peek_protocol_action.name[0]])] + suffix)
                labels[peek_protocol_action.name[0]] += 1
                resources = peek_protocol_action.nodes

                dagtask = DAGResourceSubTask(name=name, c=ceil(peek_protocol_action.duration / slot_size),
                                             parents=parent_tasks[peek_protocol_action.name],
                                             dist=peek_protocol_action.dist, resources=resources)

                # Record fidelity and rate in task description
                F = round(peek_protocol_action.F, 2)
                R = round(peek_protocol_action.R, 2)
                dagtask.description = "F={}, R={}".format(F, R)

                # Record the parent/child relationship between this subtask and others
                for parent in parent_tasks[peek_protocol_action.name]:
                    parent.add_child(dagtask)

                tasks.append(dagtask)

                if peek_child_action:
                    parent_tasks[peek_child_action.name].append(dagtask)

                last_action, _ = stack.pop()

    # Construct the overal DAGTask representing the repeater protocol
    source, dest, fidelity, rate = request
    task_id = randint(0, 100)
    task_name = "S={}, D={}, F={}, R={}, ID={}".format(source, dest, fidelity, rate, task_id)
    main_dag_task = PeriodicBudgetResourceDAGTask(name=task_name, tasks=tasks, p=ceil(1 / rate / slot_size), k=100)
    return main_dag_task


def schedule_dag_for_resources(dagtask, topology):
    """
    Schedules a protocol DAGTask onto network resources. Performs ASAP, then ALAP, then shifts swaps and distillations
    as early as possible
    :param dagtask: type DAGTask
        The DAGTask representing the protocol to map to resources
    :param topology: tuple
        Tuple of networkx.Graphs that represent the communication and connectivity graphs of the quantum network
    :return: type tuple
        Tuple of the scheduled DAGTask, the cumulative amount of time that links decohered, and a boolean representing
        whether the mapping is correct or not
    """
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
    """
    Schedules a protocol DAGTask to network resources in ASAP fashion
    :param dagtask: type DAGTask
        The DAGTask representing the protocol to be scheduled
    :param topology: type tuple
        Tuple of networkx.Graphs that represent the communication resources and connectivity information o the quantum
        network
    :return:
    """
    # Get the node resources
    comm_q_topology, node_topology = topology
    nodes = dagtask.resources
    node_comm_resources = dict([(n, node_topology.nodes[n]["comm_qs"]) for n in nodes])
    node_storage_resources = dict([(n, node_topology.nodes[n]["storage_qs"]) for n in nodes])
    comm_to_storage_resources = dict(
        [(comm_name, node_storage_resources[n]) for n in nodes for comm_name in node_topology.nodes[n]["comm_qs"]])
    all_node_resources = dict(
        [(n, node_topology.nodes[n]["comm_qs"] + node_topology.nodes[n]["storage_qs"]) for n in nodes])

    # Create  aschedule of the resources used
    resource_schedules = defaultdict(list)

    # Use a stack for post-order traversal
    stack = []

    # Start traversal with the last protocol operation
    [sink] = dagtask.sinks
    task = sink
    last_task = None

    # Sort each subtask's parents by their distance from the sink and then by name
    for task in dagtask.subtasks:
        task.parents = list(sorted(task.parents, key=lambda pt: (pt.dist, pt.name)))

    # Map the protocol in post-order traversal
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
                # Get the available resources for this operation
                possible_task_resources = get_possible_resources_for_task(peek_task, node_comm_resources,
                                                                          all_node_resources)
                # Schedule operation asap using some set of the resources
                schedule_task_asap(peek_task, possible_task_resources, resource_schedules, comm_to_storage_resources)
                last_task = stack.pop()

    # Record all the resources used
    dagtask.resources = list(set([r for subtask in dagtask.subtasks for r in subtask.resources]))

    # Compute the latency
    asap_latency = get_dag_exec_time(dagtask)

    # Create schedules for each resource used based on the DAGTask
    resource_schedules = defaultdict(list)
    for task in dagtask.subtasks:
        slots = [(task.a + i, task) for i in range(ceil(task.c))]
        for resource in task.resources:
            resource_schedules[resource] += slots

    # Sort each schedule
    for resource in resource_schedules.keys():
        resource_schedules[resource] = list(sorted(resource_schedules[resource]))

    # Get the latency and verify the scheduling
    asap_decoherence = get_schedule_decoherence(dagtask.get_resource_schedules(), asap_latency)
    asap_correct = verify_dag(dagtask)
    return asap_latency, asap_decoherence, asap_correct


def get_possible_resources_for_task(task, node_comm_resources, all_node_resources):
    """
    Obtains the resources that may be used for the task.
    :param task: type DAGSubTask
        Task representing a protocol action to map to resource
    :param node_comm_resources: type dict
        Dictionary of node to communication resource identifiers
    :param all_node_resources: type dict
        Dictionary of node to all both communication and storage resources
    :return: type list
        List of the resources that may be used for the task
    """
    # For link generation we need to check for any unused resources
    if task.name[0] == "L":
        possible_task_resources = [node_comm_resources[n] for n in task.resources]

    # For distill or swap, we need to reuse resources from previous operations
    else:
        possible_task_resources = get_resources_from_parents(task, all_node_resources)

    return possible_task_resources


def get_resources_from_parents(task, all_node_resources):
    """
    Searches the subtrees the incoming task depends on for resources to use. Used for finding resources for
    entanglement swaps and entanglement distillations.
    :param task: type DAGSubTask
        The task to find resources for
    :param all_node_resources: type dict
        Dictionary of nodes to network resources
    :return: type list
        List of resources available to the node at the task
    """
    possible_task_resources = []
    task_nodes = task.resources

    # Search backward through parents to find resources holding links
    for pt in task.parents:
        # If we are looking at a swap operation, need to go further back to find resources holding the link after swap
        if pt.name[0] == "S":
            q = [pt]
            while q:
                search_task = q.pop(0)
                # Add parents of swap operations
                if search_task.name[0] == "S":
                    for p in search_task.parents:
                        q.append(p)
                # Otherwise check this task
                else:
                    possible_task_resources += get_resources_for_task_nodes(search_task, task_nodes, all_node_resources)
        else:
            possible_task_resources += get_resources_for_task_nodes(pt, task_nodes, all_node_resources)

    return possible_task_resources


def get_resources_for_task_nodes(task, task_nodes, all_node_resources):
    """
    Obtains the resources available at task's nodes from the set of task_nodes
    :param task: type DAGSubTask
        The task to obtain resources for
    :param task_nodes: type list
        List of DAGSubTasks to obtain resources from
    :param all_node_resources: type dict
        Dictionary of nodes to network resources
    :return:
    """
    if task.name[0] == "D":
        return [[r] for r in list(sorted(task.locked_resources)) for tn in task_nodes if r in all_node_resources[tn]]
    elif task.name[0] == "L":
        return [[r] for tn in task_nodes for r in task.locked_resources if r in all_node_resources[tn]]
    else:
        raise Exception()


def schedule_task_asap(task, task_resources, resource_schedules, storage_resources):
    """
    Schedules a task onto the node resources as soon as possible.
    :param task: type DAGSubTask
        The task to schedule onto resources
    :param task_resources: type list
        List of resources that may be used to execute the task
    :param resource_schedules: type dict
        Dictionary of resource identifiers to the time periods where they are occupied by other tasks
    :param storage_resources: type dict
        Dictionary of storage resources to the communication resource they may interact with for storing a link
    :return: type list
        A list of the resources that are used for the task
    """
    # Track which mapping can be scheduled earliest
    earliest_possible_start = float('inf')
    earliest_resources = None
    earliest_mapping = None

    # Iterate over combinations of resources to use
    for resource_set in list(itertools.product(*task_resources)):
        # Find the earliest time the task could start based on when parents finished
        earliest_start = max([0] + [p.a + ceil(p.c) for p in task.parents if set(resource_set) & set(p.resources)])

        # Starting from earliest start, find the first possible time slot the operation can start
        possible_start, locked_storage_mapping = get_earliest_start_for_resources(earliest_start, task, resource_set,
                                                                                  resource_schedules, storage_resources)

        # Record the earliest start time and resources
        if earliest_start <= possible_start < earliest_possible_start and \
                (earliest_mapping is None or len(earliest_mapping.keys()) >= len(locked_storage_mapping.keys())):
            earliest_possible_start = possible_start
            earliest_resources = resource_set
            earliest_mapping = locked_storage_mapping

        # If we found a set of resources that agree with earliest possible starting point, use them
        if possible_start == earliest_start:
            break

    # Report if we couldn't schedule the operation
    if earliest_possible_start == float('inf') and task.name[0] == "L":
        logger.debug("Failed to schedule {}".format(task.name))
        return None

    # Record the slots that this task is in progress
    earliest_resources = list(set(earliest_resources))
    slots = []
    for slot in range(earliest_possible_start, earliest_possible_start + ceil(task.c)):
        slots.append((slot, task))

    # Apply the storage resource mapping for the operation
    sr_mapping = defaultdict(lambda: (float('inf'), None))
    if task.name[0] == "L" and earliest_mapping == {}:
        # Iterate over resources
        for lr in earliest_resources:
            rs = resource_schedules[lr]

            # Skip resource if locked
            if rs:
                last_slot, last_task = rs[-1]
                if task_locks_resource(last_task, [lr]):
                    continue

            # Iterate over storage resources
            for sr in storage_resources[lr]:
                rs = resource_schedules[sr]
                # Check if resource available and we can assign it to this operation
                if rs:
                    last_slot, last_task = rs[-1]
                    earliest_slot, _ = sr_mapping[lr]
                    if not task_locks_resource(last_task, [sr]) and last_slot + 1 < earliest_slot:
                        sr_mapping[lr] = (last_slot, sr)

                elif -1 < sr_mapping[lr][0]:
                    sr_mapping[lr] = (-1, sr)

            # Get the last slot the resource was used
            rs = resource_schedules[lr]
            if rs:
                last_slot, last_task = rs[-1]
            else:
                last_slot = 0

            # Track the last slot and associated resource with storage resource
            if sr_mapping[lr][1] is None:
                sr_mapping[lr] = (last_slot, lr)

        # Record the storage resource in the set of all resources used
        for t, sr in sr_mapping.values():
            if sr is not None and t + 1 <= earliest_possible_start:
                earliest_resources.append(sr)

    # Record the slots that these resources are occupied for
    for r in list(set(earliest_resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    # If this is a link generation operation, track which resources get locked due to
    # link storage
    task.a = earliest_possible_start
    task.resources = list(sorted(set(earliest_resources)))
    if task.name[0] == "L" and earliest_mapping == {}:
        task.locked_resources = list(task.resources)
        for lr in sr_mapping.keys():
            if sr_mapping[lr][1] != lr and sr_mapping[lr][1] in earliest_resources:
                task.locked_resources.remove(lr)

    # For distillation we always lock a resource for storing the distilled link
    elif task.name[0] == "D":
        task.locked_resources = task.resources[1:4:2]

    logger.debug("Scheduled {} with resources {} and locked resources {} at t={}".format(task.name, task.resources,
                                                                                         task.locked_resources, task.a))
    return list(set(earliest_resources))


def get_earliest_start_for_resources(earliest, task, resource_set, resource_schedules, storage_resources):
    """
    Finds the earliest point in time where the set of resources may be used for executing a task
    :param earliest: type int
        The earliest point in time where the task may begin
    :param task: type DAGSubTask
        The task to obtain the earliest start time for
    :param resource_set: type list
        List of resources to use for task.
    :param resource_schedules: type dict
        Dictionary of resource identifiers to the time periods where they are occupied by other tasks
    :param storage_resources: type dict
        Dictionary of storage resource identifiers to communication resource identifiers that can interact
    :return: type int
        The earliest slot where the task may begin using the set of resources
    """
    # Always get the first earliest pair of communication qubits and storage qubits, if no storage qubits left leave
    # the link in the comm qubits
    occupied_slots = defaultdict(list)

    # Find out when resources are not available
    for r in resource_set:
        rs = resource_schedules[r]
        for slot, t in rs:
            occupied_slots[slot].append(t)

    # If resources are available, just return the earliest time and no mapping
    if not occupied_slots:
        return earliest, {}

    else:
        # Convert occupied_slots into ranges where not all resources available
        occupied_ranges = list(sorted(to_ranges(occupied_slots.keys())))
        _, e = occupied_ranges[-1]

        # Find an immediate mapping to storage resources
        if task.name[0] == "L":
            # Track the best mapping
            sr_mapping = defaultdict(lambda: (float('inf'), None))

            # Iterate over communication resources
            for lr in resource_set:
                # Get schedule
                rs = resource_schedules[lr]

                # Check if this resource is occupied by the last task that used it
                if rs:
                    last_slot, last_task = rs[-1]
                    if task_locks_resource(last_task, [lr]):
                        return float('inf'), {}

                # Iterate over the resources that can be used for storing the link
                for sr in storage_resources[lr]:
                    # Check if the resource is locked
                    rs = resource_schedules[sr]
                    if rs:
                        last_slot, last_task = rs[-1]
                        earliest_slot, _ = sr_mapping[lr]
                        # Record storage mapping if free
                        if not task_locks_resource(last_task, [sr]) and last_slot + 1 < earliest_slot:
                            sr_mapping[lr] = (last_slot, sr)

                    # If not scheduled then always available at earliest chance
                    elif -1 < sr_mapping[lr][0]:
                        sr_mapping[lr] = (-1, sr)

                # Check when the communication resource was last used
                rs = resource_schedules[lr]
                if rs:
                    last_slot, last_task = rs[-1]
                else:
                    last_slot = 0

                # If we couldn't find a storage resource then we store in the communication resource
                if sr_mapping[lr][1] is None:
                    sr_mapping[lr] = (last_slot, lr)

            # If we found a storage resource for communication resources then find the
            # earliest time everything is available
            if all([sr is not None for _, sr in sr_mapping.values()]):
                earliest_resource_time = max([e] + [t for t, _ in sr_mapping.values()])
                return earliest_resource_time + 1, {}

        # Get the last task of each resource
        last_tasks = []
        for r in resource_set:
            rs = resource_schedules[r]
            if rs:
                last_task_slot, last_task = rs[-1]
                last_tasks.append((r, last_task))

        # Check that all resources are not locked by previous task
        if not (task.name[0] == "L" and any([task_locks_resource(t, [r]) for r, t in last_tasks])):
            if e >= earliest:
                return e + 1, {}
            else:
                return earliest, {}
        else:
            return float('inf'), {}


def to_ranges(iterable):
    """
    Converts a list of ints to a set of ranges that cover the list
    :param iterable: type list
        List of ints
    :return: type list
        List of tuples that represent ranges that cover the iterable
    """
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def task_locks_resource(task, resources):
    """
    Checks if a task locks the resources. This is used for checking if a resource is occupied storing a link between
    protocol actions
    :param task: type DAGSubTask
        The task that may/may not lock the resources
    :param resources: type list
        List of resources to check if task locks
    :return: type bool
        True/False if task locks the resources
    """
    link_lock = (task.name[0] == "L" and any([r in task.locked_resources for r in resources]))
    distill_lock = (task.name[0] == "D" and any([r in list(sorted(task.locked_resources)) for r in resources]))
    return link_lock or distill_lock


def convert_task_to_alap(dagtask):
    """
    Converts a task that was mapped to resources in ASAP fashion to ALAP fashion
    :param dagtask: type DAGTask
        The DAGTask representing the repeater protocol
    :return: None
    """
    # Last task doesn't move, initialize the queue with all subtasks
    queue = list(sorted(dagtask.subtasks, key=lambda task: -task.a))
    resource_schedules = defaultdict(list)
    last_task = queue.pop(0)

    # Obtain the schedule for each resource
    for r in last_task.resources:
        resource_schedules[r] = [(s, last_task) for s in range(last_task.a, last_task.a + ceil(last_task.c))]

    # Iteratively push tasks into the future
    while queue:
        task = queue.pop(0)

        # Find the latest t <= child.a s.t. all resources available
        schedule_task_alap(task, resource_schedules)

    # Shift all start times backward if we managed to move the start of the protocol
    # from t=0
    earliest = min([task.a for task in dagtask.subtasks])
    if earliest != 0:
        # Shift task start times back
        for task in dagtask.subtasks:
            task.a -= earliest

        # Shift resource schedules back
        for r in resource_schedules.keys():
            schedule = resource_schedules[r]
            resource_schedules[r] = [(s - earliest, t) for s, t in schedule]

    alap_latency = get_dag_exec_time(dagtask)
    alap_decoherence = get_schedule_decoherence(dagtask.get_resource_schedules(), alap_latency)
    alap_correct = verify_dag(dagtask)
    return alap_latency, alap_decoherence, alap_correct


def schedule_task_alap(task, resource_schedules):
    """
    Schedules a task to resources in ALAP fashion
    :param task: type DAGSubTask
        The task representing the protocol action to schedule
    :param resource_schedules: type dict
        Dictionary of resource identifiers to the time periods where the resource is occupied
    :return: None
    """
    # Get possible times, start with currently assigned
    possible = [task.a]

    # Get the start times of children (in the future), see how far we can push subtask forward
    child_starts = [ct.a - ceil(task.c) for ct in task.children if set(task.resources) & set(ct.resources)]
    if child_starts:
        possible += [min(child_starts)]
    latest = max(possible)

    latest = min([latest] + [resource_schedules[r][0][0] - ceil(task.c) for r in task.resources
                             if resource_schedules[r]])

    slots = [(s, task) for s in range(latest, latest + ceil(task.c))]

    # Update the set of slots in the schedule
    for r in list(set(task.resources)):
        resource_schedules[r] = list(sorted(resource_schedules[r] + slots))

    if latest != task.a:
        logger.debug("Moved task {} from t={} to t={}".format(task.name, task.a, latest))

    # Update the start time
    task.a = latest
    logger.debug("Scheduled {} with resources {} at t={}".format(task.name, task.resources, task.a))


def get_latest_slot_for_resources(latest, task, schedule_set):
    """
    Finds the latest opportunity that a task may be executed
    :param latest: type int
        A maximum bound on the latest point where a task may be executed
    :param task: type DAGSubTask
        The task to obtain the latest starting slot for
    :param schedule_set: type list
        List of occupied time slots of the resources used for the task
    :return: type int
        The latest slot where task may begin
    """
    # Obtain set of occupied slots across resources
    occupied_slots = set()
    for rs in schedule_set:
        occupied_slots |= set(rs)

    # Filter ones that are earlier than latest opportunity
    occupied_slots = list(filter(lambda s: s <= latest, list(sorted(occupied_slots))))

    # Settle for latest if nothing else found
    if not occupied_slots or occupied_slots[-1] < latest:
        return latest
    else:
        occupied_ranges = list(reversed(list(to_ranges(occupied_slots))))
        for (s1, e1), (s2, e2) in zip(occupied_ranges, occupied_ranges[1:]):
            if s1 - e2 >= task.c:
                return e2 + 1

        return occupied_ranges[-1][0] - ceil(task.c)


def shift_distillations_and_swaps(dagtask):
    """
    Shifts the tasks for entanglement swap and entanglement distillation ASAP
    :param dagtask: type DAGTask
        The task representing the repeater protocol to modify
    :return: None
    """
    # Get the resource schedules for link operations
    resource_schedules = defaultdict(IntervalTree)
    for link_task in dagtask.sources:
        begin = link_task.a
        end = begin + ceil(link_task.c)
        interval = Interval(begin, end, link_task)
        for resource in link_task.resources:
            resource_schedules[resource].add(interval)

    # Iterate over distillation and swap operations
    for task in sorted(dagtask.subtasks, key=lambda task: task.a):
        if task.name[0] == "D" or task.name[0] == "S":
            # Find when the latest parent operation ends
            parent_ends = [p.a + ceil(p.c) for p in task.parents if set(p.resources) & set(task.resources)]
            parent_task_end = max(parent_ends) if parent_ends else 0

            # Check resource availability
            resource_availabilities = []
            for resource in task.resources:
                interval_set = resource_schedules[resource].envelop(0, task.a)
                if interval_set:
                    interval = sorted(interval_set)[-1]
                    resource_availabilities.append(interval.end)

            # Get the earliest point these operations can be shifted to
            earliest_start = max([parent_task_end] + resource_availabilities)
            task.a = earliest_start
            if task.c > 0:
                interval = Interval(begin=task.a, end=task.a + ceil(task.c), data=task)
                for resource in task.resources:
                    resource_schedules[resource].add(interval)

    # Update the schedules with the shifted slots
    resource_schedules_new = {}
    for resource in resource_schedules.keys():
        resource_schedule = []
        for interval in resource_schedules[resource]:
            task = interval.data
            slots = [(task.a + i, task) for i in range(ceil(task.c))]
            resource_schedule += slots
        resource_schedules_new[resource] = list(sorted(resource_schedule))

    # Collect some stats and make sure dag still correct
    shift_latency = get_dag_exec_time(dagtask)
    shift_decoherence = get_schedule_decoherence(dagtask.get_resource_schedules(), shift_latency)
    shift_correct = verify_dag(dagtask)

    return shift_latency, shift_decoherence, shift_correct


def verify_dag(dagtask, node_resources=None):
    """
    Verifies whether the repeater protocol in the DAGTask is correct
    :param dagtask: type DAGTask
        Task representing the repeater protocol
    :param node_resources: type list
        List of the resources held across the nodes over which the task executes
    :return: type bool
        True/False if the DAGTask is correct
    """
    resource_intervals = defaultdict(IntervalTree)
    valid = True
    for subtask in dagtask.subtasks:
        # Check that the subtask is not using the same resources after the child has started
        for child in subtask.children:
            if child.a < subtask.a + ceil(subtask.c) and set(child.resources) & set(subtask.resources):
                valid = False

        # For link generation, check that
        # 1) if the subtask is link generation, that it only locks 2 resources
        # 2) The number of resources used by the task is at least 2/3/4 depending on
        # whether storage resources were available
        if subtask.name[0] == "L" and \
                (len(set(subtask.locked_resources)) != 2 or len(set(subtask.resources)) not in [2, 3, 4]):
            logger.error("Link generation subtask has incorrect set of resources")
            valid = False

        # Check that for a swap task the number of resource used is only 2 (one qubit for each link)
        if subtask.name[0] == "S" and len(set(subtask.resources)) != 2:
            logger.error("Swapping subtask has incorrect set of resources")
            valid = False

        # Check that for a distillation task the number of resources used is 4 (2 qubits for each link)
        # and that the number of locked resources is 2 (the qubits storing the link)
        if subtask.name[0] == "D" and (len(set(subtask.resources)) != 4 or len(set(subtask.locked_resources)) != 2):
            logger.error("Distillation subtask has incorrect set of resources")
            valid = False

        # Check the resources used by tasks
        if subtask.c > 0:
            subtask_interval = Interval(subtask.a, subtask.a + subtask.c, subtask)
            for resource in subtask.resources:
                # First check that the resource is actually one of the valid ones
                if node_resources and (resource not in node_resources):
                    continue

                # Check if the current subtask's usage of the resource overlaps with
                # another subtask
                if resource_intervals[resource].overlap(subtask_interval.begin, subtask_interval.end):
                    overlapping = sorted(resource_intervals[resource][subtask_interval.begin:subtask_interval.end])[0]
                    logger.error("Subtask {} overlaps at resource {}"
                                 " during interval {},{} with task {}".format(subtask.name, resource, overlapping.begin,
                                                                              overlapping.end, overlapping.data.name))
                    valid = False

                # Track this subtask's use of this resource
                resource_intervals[resource].add(subtask_interval)

    # Check that links are produced then consumed between subtasks
    # Iterate over each resource
    for resource in resource_intervals.keys():
        sorted_intervals = sorted(resource_intervals[resource])

        # Check each pair of consecutive subtasks using the resource
        for iv1, iv2 in zip(sorted_intervals, sorted_intervals[1:]):
            t1 = iv1.data
            t2 = iv2.data

            # Two link generations cannot store a link in the same resource consecutively
            if t1.name[0] == "L" and t2.name == "L":
                logger.error("Consecutive link generation subtasks on common resource")
                valid = False

            # Check that a link generation is not trying to use a locked resource from
            # distillation
            elif t1.name[0] == "D" and t2.name == "L" and any([r in t1.locked_resources for r in t2.locked_resources]):
                logger.error("Distillation followed by link generation on same resource")
                valid = False

    return valid


def get_schedule_decoherence(resource_schedules, completion_time):
    """
    Obtains the number of time slots where links sits idle between protocol actions
    :param resource_schedules: type dict
        Dictionary of resource identifiers to time periods where the resource is occupied
    :param completion_time: type int
        The time slot where the protocol is completed
    :return: type int
        The number of time slots where links are decohering
    """
    total_decoherence = 0
    resource_decoherences = {}
    for r in sorted(resource_schedules.keys()):
        resource_decoherence = 0
        rs = resource_schedules[r]

        # Occupancy periods represent storing links, just count these slots
        for slot, task in rs:
            if task.name[0] == "O":
                resource_decoherence += 1

        resource_decoherences[r] = resource_decoherence
        total_decoherence += resource_decoherence

    return total_decoherence

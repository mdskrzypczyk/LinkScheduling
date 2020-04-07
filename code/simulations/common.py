import networkx as nx
import random
from math import ceil
from collections import defaultdict
from jobscheduling.protocolgen import create_protocol, LinkProtocol, DistillationProtocol, SwapProtocol
from jobscheduling.protocols import convert_protocol_to_task, schedule_dag_for_resources
from jobscheduling.schedulers.BlockPBEDF import UniResourcePreemptionBudgetScheduler,\
    UniResourceFixedPointPreemptionBudgetScheduler, UniResourceConsiderateFixedPointPreemptionBudgetScheduler
from jobscheduling.schedulers.SearchBlockPBEDF import MultipleResourceConsiderateBlockPreemptionBudgetScheduler,\
    MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler, \
    MultipleResourceConsiderateSegmentPreemptionBudgetScheduler
from jobscheduling.schedulers.CEDF import UniResourceCEDFScheduler, MultipleResourceBlockCEDFScheduler
from jobscheduling.schedulers.BlockNPEDF import UniResourceBlockNPEDFScheduler, MultipleResourceBlockNPEDFScheduler
from jobscheduling.schedulers.NPEDF import MultipleResourceNonBlockNPEDFScheduler


def get_schedulers():
    schedulers = [
        UniResourceConsiderateFixedPointPreemptionBudgetScheduler,
        UniResourceBlockNPEDFScheduler,
        UniResourceCEDFScheduler,
        MultipleResourceBlockCEDFScheduler,
        MultipleResourceBlockNPEDFScheduler,
        MultipleResourceNonBlockNPEDFScheduler,
        MultipleResourceConsiderateBlockPreemptionBudgetScheduler,
        MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler,
        MultipleResourceConsiderateSegmentPreemptionBudgetScheduler,
    ]
    return schedulers


def get_network_demands(network_topology, num):
    _, nodeG = network_topology
    demands = []
    end_nodes = [node for node in nodeG.nodes if nodeG.nodes[node]["end_node"] is True]
    for num_demands in range(num):
        src, dst = random.sample(end_nodes, 2)
        fidelity = round(0.6 + random.random() * (3 / 10), 3)                    # Fidelity range between F=0.6 and 1
        rate = 10 / (2**random.choice([i for i in range(7, 11)]))       # Rate range between 0.2 and 1
        demands.append((src, dst, fidelity, rate))
    return demands


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


def select_rate(achieved_rate, slot_size):
    rates = [32 / (2 ** i) for i in range(10)]  # Rate range between
    rates = list(filter(lambda r: slot_size < r < achieved_rate, rates))
    if rates:
        return random.choice(rates)
    else:
        return 0


def get_network_resources(topology):
    _, nodeG = topology
    network_resources = {}
    network_nodes = nodeG.nodes
    for node in network_nodes:
        try:
            numCommResources = len(nodeG.nodes[node]['comm_qs'])
            numStorResources = len(nodeG.nodes[node]['storage_qs'])
            network_resources[node] = {
                "comm": numCommResources,
                "storage": numStorResources,
                "total": numCommResources + numStorResources
            }
        except:
            import pdb
            pdb.set_trace()
    return network_resources


def get_resource_string(resource):
    resource_node, resource_id = resource.split('-')
    resource_type = resource_id[0]
    return resource_node + resource_type


def balance_taskset_resource_utilization(taskset, node_resources):
    resource_utilization = {}
    resource_types = defaultdict(set)
    for node in node_resources:
        comm_qs = node_resources[node]["comm_qs"]
        for c in comm_qs:
            resource_utilization[c] = 0
            rt = get_resource_string(c)
            resource_types[rt] |= {c}
        storage_qs = node_resources[node]["storage_qs"]
        for s in storage_qs:
            resource_utilization[s] = 0
            rt = get_resource_string(s)
            resource_types[rt] |= {s}

    for task in taskset:
        task_period = task.p
        task_resource_utilization = defaultdict(float)
        task_resource_types = defaultdict(set)
        task_resource_intervals = task.get_resource_intervals()
        for resource, itree in task_resource_intervals.items():
            total_occupation_time = sum([i.end - i.begin for i in itree])
            task_resource_utilization[resource] += total_occupation_time / task_period
            rt = get_resource_string(resource)
            task_resource_types[rt] |= {resource}

        resource_mapping = {}
        for rt, resources in task_resource_types.items():
            tru = list(sorted([(task_resource_utilization[r], r) for r in task_resource_types[rt]]))
            ru = list(sorted([(-resource_utilization[r], r) for r in resource_types[rt]]))
            for (vu, vr), (_, pr) in zip(tru, ru):
                resource_mapping[vr] = pr
                resource_utilization[pr] += vu

        for subtask in task.subtasks:
            new_resources = []
            for vr in subtask.resources:
                new_resources.append(resource_mapping[vr])

            new_locked_resources = []
            for vr in subtask.locked_resources:
                new_locked_resources.append(resource_mapping[vr])

            subtask.resources = new_resources
            subtask.locked_resources = new_locked_resources

        new_task_resources = []
        for vr in task.resources:
            new_task_resources.append(resource_mapping[vr])
        task.resources = new_task_resources


def check_resource_utilization(taskset):
    resource_utilization = defaultdict(float)
    for task in taskset:
        resource_intervals = task.get_resource_intervals()
        for resource, itree in resource_intervals.items():
            utilization = sum([i.end - i.begin for i in itree]) / task.p
            resource_utilization[resource] += utilization
            if resource_utilization[resource] > 1:
                logger.warning("Taskset overutilizes resource {}, computed {}".format(resource, resource_utilization[resource],))
                return False

    for resource in resource_utilization.keys():
        resource_utilization[resource] = round(resource_utilization[resource], 3)

    logger.info("Resource utilization: {}".format(resource_utilization))
    return True


def get_taskset(num_tasks, fidelity, topology, slot_size):
    taskset = []
    num_succ = 0
    while len(taskset) < num_tasks:
        s, d = random.sample(['0', '2', '6', '8'], 2)
        rate = 10 / (2**random.choice([i for i in range(3, 7)]))
        demand = (s, d, fidelity, rate)
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

            new_rate = select_rate(achieved_rate, slot_size)
            if new_rate == 0:
                logger.warning("Could not provide rate for {}".format(demand))
                continue
            scheduled_task.p = ceil(1 / new_rate / slot_size)

            s, d, f, r = demand
            demand = (s, d, f, new_rate)
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

    balance_taskset_resource_utilization(taskset, node_resources=topology[1].nodes)
    return taskset
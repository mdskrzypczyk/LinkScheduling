import networkx as nx
import random
import time
from math import ceil
from device_characteristics.nv_links import load_link_data
from jobscheduling.log import LSLogger
from jobscheduling.protocols import convert_protocol_to_task, schedule_dag_for_resources
from jobscheduling.task import get_lcm_for
from simulations.common import load_results, write_results, get_schedulers, get_balanced_taskset, schedule_taskset, \
    get_protocol_without_rate_constraint, balance_taskset_resource_utilization

logger = LSLogger()


def gen_star_topology1(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3,
                       link_length=5):
    num_nodes = 4
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_length)]
    # Line
    Gcq = nx.Graph()
    G = nx.Graph()

    # First make the center
    comm_qs = []
    storage_qs = []
    i = num_nodes - 1
    for c in range(num_rep_comm_q):
        comm_q_id = "{}-C{}".format(i, c)
        comm_qs.append(comm_q_id)
    for s in range(num_rep_storage_q):
        storage_q_id = "{}-S{}".format(i, s)
        storage_qs.append(storage_q_id)
    Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
    G.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs, end_node=False)

    # Then make the end nodes
    for i in range(num_nodes - 1):
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        G.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs, end_node=True)

        center_node_id = num_nodes - 1
        for j in range(num_rep_comm_q):
            for k in range(num_end_node_comm_q):
                Gcq.add_edge("{}-C{}".format(center_node_id, j), "{}-C{}".format(i, k))

        G.add_edge("{}".format(center_node_id), "{}".format(i), capabilities=link_capability, weight=link_length)

    return Gcq, G


def gen_star_topology2(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3,
                       link_length=5):
    num_nodes = 4
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_length)]
    # Line
    Gcq = nx.Graph()
    G = nx.Graph()

    # First make the center
    comm_qs = []
    storage_qs = []
    i = num_nodes - 1

    for c in range(num_rep_comm_q, 2 * num_rep_comm_q):
        comm_q_id = "{}-C{}".format(i, c)
        comm_qs.append(comm_q_id)

    for s in range(num_rep_storage_q, 2 * num_rep_storage_q):
        storage_q_id = "{}-S{}".format(i, s)
        storage_qs.append(storage_q_id)

    Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
    G.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs, end_node=False)

    # Then make the end nodes
    for i in range(num_nodes - 1):
        comm_qs = []
        storage_qs = []

        for c in range(num_end_node_comm_q, 2 * num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(i, c)
            comm_qs.append(comm_q_id)

        for s in range(num_end_node_storage_q, 2 * num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(i, s)
            storage_qs.append(storage_q_id)

        Gcq.add_nodes_from(comm_qs, node="{}".format(i), storage=storage_qs)
        G.add_node("{}".format(i), comm_qs=comm_qs, storage_qs=storage_qs, end_node=True)

        center_node_id = num_nodes - 1
        for j in range(num_rep_comm_q):
            for k in range(num_end_node_comm_q):
                Gcq.add_edge("{}-C{}".format(center_node_id, j), "{}-C{}".format(i, k))

        G.add_edge("{}".format(center_node_id), "{}".format(i), capabilities=link_capability, weight=link_length)

    return Gcq, G


def get_full_taskset(taskset1, taskset2, topology, slot_size):
    full_taskset = []
    proc = []
    for task1, task2 in zip(taskset1, taskset2):
        proc.append(task1)
        proc.append(task2)

    proc += taskset1[len(taskset2):] if len(taskset1) > len(taskset2) else taskset2[len(taskset1):]
    for task in proc:
        items = task.name.split(" ")
        source = items[0].strip(",").split("=")[-1]
        destination = items[1].strip(",").split("=")[-1]
        fidelity = float(items[2].strip(",").split("=")[-1])
        rate = float(items[3].strip(",").split("=")[-1])
        demand = (source, destination, fidelity, 1)
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

            demand = (source, destination, fidelity, rate)
            scheduled_task.name = "S={}, D={}, F={}, R={}, ID={}".format(*demand, random.randint(0, 100))
            scheduled_task.p = ceil(1 / rate / slot_size)
            asap_dec, alap_dec, shift_dec = decoherence_times
            logger.info("Results for {}:".format(demand))
            logger.info(
                "Successfully created protocol and task for demand (S={}, D={}, F={}, R={})".format(*demand))
            full_taskset.append(scheduled_task)

        except Exception as err:
            logger.exception("Error occurred while generating tasks: {}".format(err))

        balance_taskset_resource_utilization(full_taskset, node_resources=topology[1].nodes)

    return full_taskset


def main():
    fidelities = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    full_topology = gen_star_topology1(num_end_node_comm_q=2, num_end_node_storage_q=6, num_rep_comm_q=2,
                                       num_rep_storage_q=6)
    half_topology1 = gen_star_topology1(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1,
                                        num_rep_storage_q=3)
    half_topology2 = gen_star_topology2(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1,
                                        num_rep_storage_q=3)
    slot_size = 0.01
    schedulers = get_schedulers()
    results_file = "resource_results/resource_results_{}.json"
    num_results = 0
    while num_results < 100:
        print("Starting new run {}".format(num_results))

        run_results = {}
        for fidelity in fidelities:
            fidelity_data = {}
            print("Running fidelity {}".format(fidelity))
            print("Generating taskset")
            taskset1 = get_balanced_taskset(half_topology1, fidelity, slot_size)
            taskset2 = get_balanced_taskset(half_topology2, fidelity, slot_size)
            half_taskset = taskset1 + taskset2
            full_taskset = get_full_taskset(taskset1, taskset2, full_topology, slot_size)
            for taskset_name, taskset in zip(["full", "half"], [full_taskset, half_taskset]):
                print("Completed generating {} taskset of size {}".format(taskset_name, len(taskset)))
                print("Hyperperiod: {}".format(get_lcm_for([task.p for task in taskset])))
                for scheduler_class in schedulers:
                    scheduler = scheduler_class()
                    scheduler_key = type(scheduler).__name__ + "_{}".format(taskset_name)
                    print("Running scheduler {}".format(scheduler_key))
                    scheduler_results = schedule_taskset(scheduler, taskset, full_topology, slot_size)
                    fidelity_data[scheduler_key] = scheduler_results

            run_results[fidelity] = fidelity_data

        run_key = time.strftime("%b %d %Y %H:%M:%S", time.gmtime(time.time()))
        results = load_results(results_file.format(run_key))
        results[run_key] = run_results
        try:
            write_results(results_file.format(run_key), results)
        except Exception:
            import pdb
            pdb.set_trace()
        print("Completed run {}".format(num_results))
        num_results += 1


if __name__ == "__main__":
    main()

import networkx as nx
import time
from device_characteristics.nv_links import load_link_data
from jobscheduling.task import get_lcm_for
from simulations.common import load_results, write_results, get_schedulers, get_balanced_taskset, schedule_taskset


def gen_plus_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3, link_length=5):
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

        G.add_edge("{}".format(center_node_id), "{}".format(i), capabilities=link_capability,
                       weight=link_length)

    return Gcq, G


def main():
    fidelities = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    topology = gen_plus_topology(num_rep_comm_q=2, num_rep_storage_q=4)
    slot_size = 0.01
    schedulers = get_schedulers()
    results_file = "plus_results/plus_rep_2c4s_results_{}.json"
    num_results = 0
    while num_results < 100:
        print("Starting new run {}".format(num_results))

        run_results = {}
        for fidelity in fidelities:
            fidelity_data = {}
            print("Running fidelity {}".format(fidelity))
            print("Generating taskset")
            taskset = get_balanced_taskset(topology, fidelity, slot_size)
            print("Completed generating taskset of size {}".format(len(taskset)))
            print("Hyperperiod: {}".format(get_lcm_for([task.p for task in taskset])))
            for scheduler_class in schedulers:
                scheduler = scheduler_class()
                scheduler_key = type(scheduler).__name__
                print("Running scheduler {}".format(scheduler_key))
                scheduler_results = schedule_taskset(scheduler, taskset, topology, slot_size)
                fidelity_data[scheduler_key] = scheduler_results

            run_results[fidelity] = fidelity_data

        run_key = time.strftime("%b %d %Y %H:%M:%S", time.gmtime(time.time()))
        results = load_results(results_file.format(run_key))
        results[run_key] = run_results
        try:
            write_results(results_file.format(run_key), results)
        except:
            import pdb
            pdb.set_trace()
        print("Completed run {}".format(num_results))
        num_results += 1


if __name__ == "__main__":
    main()
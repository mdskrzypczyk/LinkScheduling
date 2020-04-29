import networkx as nx
import time
from device_characteristics.nv_links import load_link_data
from jobscheduling.task import get_lcm_for
from simulations.common import load_results, write_results, get_schedulers, get_balanced_taskset, schedule_taskset


def gen_H_topology(num_end_node_comm_q=1, num_end_node_storage_q=3, num_rep_comm_q=1, num_rep_storage_q=3,
                   link_length=5):
    d_to_cap = load_link_data()
    link_capability = d_to_cap[str(link_length)]
    # Line
    Gcq = nx.Graph()
    G = nx.Graph()

    end_nodes = ['0', '2', '3', '5']
    repeater_nodes = ['1', '4']
    edges = [
        ('0', '1'),
        ('1', '2'),
        ('1', '4'),
        ('3', '4'),
        ('4', '5')
    ]

    for node in end_nodes:
        comm_qs = []
        storage_qs = []
        for c in range(num_end_node_comm_q):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(num_end_node_storage_q):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        G.nodes[node]["end_node"] = True

    for node in repeater_nodes:
        comm_qs = []
        storage_qs = []
        for c in range(num_rep_comm_q):
            comm_q_id = "{}-C{}".format(node, c)
            comm_qs.append(comm_q_id)
        for s in range(num_rep_storage_q):
            storage_q_id = "{}-S{}".format(node, s)
            storage_qs.append(storage_q_id)
        Gcq.add_nodes_from(comm_qs, node="{}".format(node), storage=storage_qs)
        G.add_node(node, comm_qs=comm_qs, storage_qs=storage_qs)
        G.nodes[node]["end_node"] = False

    for node1, node2 in edges:
        G.add_edge("{}".format(node1), "{}".format(node2), capabilities=link_capability, weight=link_length)

    for node1, node2 in edges:
        num_comm_node1 = num_end_node_comm_q if G.nodes[node1]["end_node"] else num_rep_comm_q
        num_comm_node2 = num_end_node_comm_q if G.nodes[node2]["end_node"] else num_rep_comm_q
        for j in range(num_comm_node1):
            for k in range(num_comm_node2):
                Gcq.add_edge("{}-C{}".format(node1, j), "{}-C{}".format(node2, k))

    return Gcq, G


def main():
    fidelities = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    topology = gen_H_topology()
    slot_size = 0.01
    schedulers = get_schedulers()
    results_file = "H_results.json"

    results = load_results(results_file)
    while True:
        if len(results.keys()) == 34:
            print("Completed 100 runs, quitting")

        print("Starting new run")
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
        results[run_key] = run_results
        try:
            write_results(results_file, results)
        except Exception:
            import pdb
            pdb.set_trace()
        print("Completed run {}".format(len(results.keys())))


if __name__ == "__main__":
    main()

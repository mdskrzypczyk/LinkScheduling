import time
from jobscheduling.task import get_lcm_for
from jobscheduling.topology import gen_plus_topology
from simulations.common import load_results, write_results, get_schedulers, get_balanced_taskset, schedule_taskset


def main():
    fidelities = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    topology = gen_plus_topology(num_rep_comm_q=2, num_rep_storage_q=4)
    slot_size = 0.01
    schedulers = get_schedulers()
    results_file = "plus_results/plus_test_results_{}.json"
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
        except Exception:
            import pdb
            pdb.set_trace()
        print("Completed run {}".format(num_results))
        num_results += 1


if __name__ == "__main__":
    main()

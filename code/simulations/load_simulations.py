import time
from jobscheduling.task import get_lcm_for
from jobscheduling.topology import gen_symm_topology
from simulations.common import load_results, write_results, get_schedulers, get_fixed_load_taskset, schedule_taskset


def main():
    fidelities = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    network_loads = [20, 25, 30, 35, 40]
    topology = gen_symm_topology()
    slot_size = 0.01
    schedulers = get_schedulers()
    results_file = "load_results.json"

    results = load_results(results_file)
    while True:
        print("Starting new run")
        run_results = {}
        for fidelity in fidelities:
            fidelity_data = {}
            print("Running fidelity {}".format(fidelity))
            for load in network_loads:
                load_data = {}
                print("Generating taskset for load {} ebit/s".format(load))
                taskset = get_fixed_load_taskset(topology, fidelity, load, slot_size)
                task_load = sum([float(t.name.split("R=")[1].split(", ")[0]) for t in taskset])
                if task_load < load:
                    import pdb
                    pdb.set_trace()
                print("Completed generating taskset of size {}".format(len(taskset)))
                print("Hyperperiod: {}".format(get_lcm_for([task.p for task in taskset])))
                for scheduler_class in schedulers:
                    scheduler = scheduler_class()
                    scheduler_key = type(scheduler).__name__
                    print("Running scheduler {}".format(scheduler_key))
                    scheduler_results = schedule_taskset(scheduler, taskset, topology, slot_size)
                    load_data[scheduler_key] = scheduler_results

                fidelity_data[load] = load_data

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

import matplotlib.pyplot as plt
from scheduler import MultipleResourceEDFScheduler, MultipleResourceBlockEDFScheduler, \
    MultipleResourceNPEDFScheduler, MultipleResourceBlockNPEDFScheduler, \
    MultipleResourceBlockCEDFScheduler,  \
    MultipleResourceBlockEDFLBFScheduler
from collections import defaultdict


def gen_topologies(n):
    # Line

    # Ring

    # Grid

    return []


def get_schedulers():
    schedulers = [
        MultipleResourceEDFScheduler,
        MultipleResourceBlockEDFScheduler,
        MultipleResourceNPEDFScheduler,
        MultipleResourceBlockNPEDFScheduler,
        MultipleResourceBlockCEDFScheduler,
        MultipleResourceBlockEDFLBFScheduler
    ]
    return schedulers


def get_network_demands(network_topology, utilization):
    return []


def get_protocols(network_topology, demands):
    return []


def convert_protocol_to_task(protocol):
    return None


def main():
    num_network_nodes = 2
    num_tasksets = 1
    utilizations = [0.1*i for i in range(1, 11)]           # Utilizations in increments of 0.1
    budget_allowances = [1*i for i in range(1, 11)]
    network_topologies = gen_topologies(num_network_nodes)
    network_schedulers = get_schedulers()

    schedule_validator = MultipleResourceBlockEDFLBFScheduler.check_feasible

    results = {}

    for topology in network_topologies:
        network_tasksets = defaultdict(list)

        for u in utilizations:
            for i in range(num_tasksets):
                # Generate task sets according to some utilization characteristics and preemption budget allowances
                # 1) Select S/D pairs in the network topology
                demands = get_network_demands(topology, u)

                # 2) Select protocol for each S/D pair
                protocols = get_protocols(topology, demands)

                # 3) Convert to task representation
                taskset = []
                for protocol in protocols:
                    taskset.append(convert_protocol_to_task(protocol))

                network_tasksets[u].append(taskset)

        # Use all schedulers
        for scheduler in network_schedulers:
            results_key = str(type(scheduler))
            scheduler_results = defaultdict(int)

            for u in utilizations:
                # Run scheduler on all task sets
                for taskset in network_tasksets[u]:
                    schedule, _ = scheduler.schedule_tasks(taskset)
                    valid = schedule_validator(schedule, taskset)

                    # Record success
                    if valid:
                        scheduler_results[u] += 1

                scheduler_results[u] /= len(network_tasksets[u])

            results[results_key] = scheduler_results

    # Plot schedulability ratio vs. utilization for each task set
    for scheduler_type, scheduler_results in results.items():
        xdata = utilizations
        ydata = [scheduler_results[u] for u in utilizations]
        plt.plot(xdata, ydata, label=scheduler_type)

    plt.show()

    # Plot schedulability ratio vs. budget allowances for each task set
    for scheduler_type, scheduler_results in results.items():
        xdata = budget_allowances
        ydata = [scheduler_results[b] for b in budget_allowances]
        plt.plot(xdata, ydata, lable=scheduler_type)

    plt.show()


if __name__ == "__main__":
    main()

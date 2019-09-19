import pdb
import numpy as np
from time import time
from schedule import Schedule, LinkSchedule
from compute_schedule_fidelity import brute_force_optimal_schedule, heuristic_search_schedule, LinkJob


def print_schedules(machine_schedules):
    print("Job/Availability Views:")
    for sched in machine_schedules:
        print(sched.get_job_view())

    for sched in machine_schedules:
        print(sched.get_availability_view())


def get_link_availability(schedule1, schedule2):
    av1 = schedule1.get_availability_view()
    av2 = schedule2.get_availability_view()
    num_slots = schedule1.get_num_total_slots()
    slot_size = schedule1.get_slot_size()
    slots = []
    for i in range(num_slots):
        if av1[i] == schedule1.SLOT_AVAILABLE and av2[i] == schedule2.SLOT_AVAILABLE:
            slots.append((np.round(i*slot_size, decimals=2), np.round(i*slot_size, decimals=2)))

    return slots


def demo_network_parameters():
    # Number of nodes
    num_machines = 4

    # Set of paths for building long distance entanglement
    paths = [
        [0, 1, 2],
        [1, 2, 3],
        [0, 3],
        [1, 0, 3]
    ]

    links = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3)
    ]

    # Decoherence weights for comm/storage qubits
    machine_params = [(0.68526781, 1.00004608) for _ in range(num_machines)]

    # (initial fidelity, expected generation time) for each link
    link_params = dict([(l, (0.9, 0.1)) for l in links])

    return num_machines, machine_params, links, link_params, paths


def H_network_parameters():
    # Number of nodes
    num_machines = 6

    # Set of paths for building long distance entanglement
    paths = [
        [0, 1, 4, 3],
        [3, 4, 5],
        [2, 1, 4, 5],
        [0, 1, 2],
        [0, 1, 4, 5],
        [2, 1, 4, 3]
    ]

    links = [
        (0, 1),
        (1, 2),
        (1, 4),
        (3, 4),
        (4, 5)
    ]

    # Decoherence weights for comm/storage qubits
    machine_params = [(0.68526781, 1.00004608) for _ in range(num_machines)]

    # (initial fidelity, expected generation time) for each link
    link_params = dict([(l, (0.9, 0.1)) for l in links])

    return num_machines, machine_params, links, link_params, paths


def double_cross_network_parameters():
    # Number of nodes
    num_machines = 8

    # Set of paths for building long distance entanglement
    paths = [
        [1, 3, 2],
        [0, 3, 6, 7],
        [4, 6, 5],
        [1, 3, 6, 4],
        [0, 3, 2],
        [4, 6, 7]
    ]

    links = [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 6),
        (4, 6),
        (5, 6),
        (6, 7)
    ]

    # Decoherence weights for comm/storage qubits
    machine_params = [(0.68526781, 1.00004608) for _ in range(num_machines)]

    # (initial fidelity, expected generation time) for each link
    link_params = dict([(l, (0.9, 0.1)) for l in links])

    return num_machines, machine_params, links, link_params, paths


def crossing_triangle_network_parameters():
    # Number of nodes
    num_machines = 9

    # Set of paths for building long distance entanglement
    paths = [
        [0, 2, 8, 6],
        [3, 5, 8, 7],
        [1, 2, 5, 4],
        [3, 5, 2, 0],
        [7, 8, 6],
        [3, 5, 8, 7],
        [3, 5, 4]
    ]

    links = [
        (0, 2),
        (1, 2),
        (2, 5),
        (2, 8),
        (3, 5),
        (4, 5),
        (5, 8),
        (6, 8),
        (7, 8)
    ]

    # Decoherence weights for comm/storage qubits
    machine_params = [(0.68526781, 1.00004608) for _ in range(num_machines)]

    # (initial fidelity, expected generation time) for each link
    link_params = dict([(l, (0.9, 0.1)) for l in links])

    return num_machines, machine_params, links, link_params, paths


def network_simulation(num_machines, machine_params, links, link_params, paths, sched_func):
    start_time = time()

    # Schedules with 20 slots with 0.1s per slot
    machine_schedules = [Schedule(schedule_size=20, slot_size=0.1) for m in range(num_machines)]

    # Link job descriptions (assume the same for now)
    link_slots = dict([(l, get_link_availability(machine_schedules[l[0]], machine_schedules[l[1]])) for l in links])
    jobs = {}
    for nodes, link_param in link_params.items():
        F_initial, duration = link_param
        lj = LinkJob(name=nodes, ID=nodes)
        lj.set_initial_fidelity(F0=F_initial)
        jobs[nodes] = lj

    jobID = 0
    # Iterate over each path
    path_fidelities = []
    path_timespans = []
    for path in paths:
        # Get the jobs for this path
        link_jobs = [jobs[tuple(sorted((n1, n2)))] for n1, n2 in zip(path, path[1:])]

        # Get the slots for these jobs
        job_slots = [link_slots[tuple(sorted((n1, n2)))] for n1, n2 in zip(path, path[1:])]

        # Get machine parameters for this path
        path_machine_params = [machine_params[p] for p in path]

        # Compute the schedule
        fidelity, timespan, slots = sched_func(path_machine_params, link_jobs, job_slots)
        path_fidelities.append(fidelity)
        path_timespans.append(timespan)
        unique_jobs = {}
        for nodes, lj in zip(zip(path, path[1:]), link_jobs):
            nodes = tuple(sorted(nodes))
            new_job = LinkJob(name=nodes, ID=jobID)
            new_job.set_initial_fidelity(lj.get_initial_fidelity())
            jobID += 1
            unique_jobs[nodes] = new_job

        # Update the schedules of the machine with the decided path
        for nodes, s in zip(zip(path, path[1:]), slots):
            nodes = tuple(sorted(nodes))
            link_job = unique_jobs[nodes]
            start, end = s
            duration = end - start + 0.1
            n1, n2 = nodes
            machine_schedules[n1].insert_job(link_job, start, duration)
            machine_schedules[n2].insert_job(link_job, start, duration)

        link_slots = dict([(l, get_link_availability(machine_schedules[l[0]], machine_schedules[l[1]])) for l in links])

    end_time = time()
    print("Computing link schedules took {}s".format(end_time-start_time))
    print_schedules(machine_schedules)
    return machine_schedules, path_fidelities, path_timespans


def main():
    network_params = crossing_triangle_network_parameters()
    msbf, pfbf, ptbf = network_simulation(*network_params, brute_force_optimal_schedule)
    msh, pfh, pth = network_simulation(*network_params, heuristic_search_schedule)
    if [ms.get_job_view() for ms in msbf] == [ms.get_job_view() for ms in msh]:
        print("Brute force and Heuristic comptue the same schedule!")


if __name__ == '__main__':
    main()

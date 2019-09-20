import pdb
import numpy as np
from random import randint, choice, sample
from collections import defaultdict
from time import time
from schedule import Schedule, LinkSchedule
from compute_schedule_fidelity import brute_force_optimal_schedule, brute_force_adjacent, \
    heuristic_search_schedule, heuristic_search_schedule_with_depth, LinkJob


def print_schedules(machine_schedules):
    print("Job/Availability Views:")
    for sched in machine_schedules:
        print(sched.get_job_view())

    for sched in machine_schedules:
        print(sched.get_availability_view())


def get_link_availability(schedule1, schedule2, duration):
    av1 = schedule1.get_availability_view()
    av2 = schedule2.get_availability_view()
    num_slots = schedule1.get_num_total_slots()
    slot_size = schedule1.get_slot_size()
    slots = []
    num_link_slots = int(np.round(duration / slot_size))
    cslots1 = schedule1.get_continuous_slots(min_size=num_link_slots)
    cslots2 = schedule2.get_continuous_slots(min_size=num_link_slots)
    for cslot in cslots1:
        if cslot in cslots2:
            time_slot = np.round(cslot[0]*slot_size, decimals=1), np.round(cslot[1]*slot_size, decimals=1)
            slots.append(time_slot)

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

    # (initial fidelity, expected generation time) for each link
    link_params = {
        (0, 1): (0.9, 0.1),
        (1, 2): (0.9, 0.2),
        (2, 3): (0.9, 0.1),
        (0, 3): (0.8, 0.3)
    }

    # Decoherence weights for comm/storage qubits
    machine_params = [(0.68526781, 1.00004608) for _ in range(num_machines)]

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
    # Schedules with 20 slots with 0.1s per slot
    machine_schedules = [Schedule(schedule_size=25, slot_size=0.1) for m in range(num_machines)]

    # Link job descriptions (assume the same for now)
    link_slots = dict([(l, get_link_availability(machine_schedules[l[0]], machine_schedules[l[1]], duration=link_params[l][1])) for l in links])
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
    for i, path in enumerate(paths):
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
        if fidelity >= 0.5:
            unique_jobs = {}
            for nodes, lj in zip(zip(path, path[1:]), link_jobs):
                nodes = tuple(sorted(nodes))
                new_job = LinkJob(name=[path, nodes], ID=jobID)
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

        link_slots = dict([(l, get_link_availability(machine_schedules[l[0]], machine_schedules[l[1]], duration=link_params[l][1])) for l in links])

    return machine_schedules, path_fidelities, path_timespans


def verify_schedule(machine_schedules, path_fidelities, network_params):
    paths = network_params[-1]
    expected_link_counts = defaultdict(int)
    for fidelity, path in zip(path_fidelities, paths):
        if fidelity >= 0.5:
            for link in zip(path, path[1:]):
                link = tuple(sorted(link))
                expected_link_counts[link] += 2

    scheduled_link_counts = defaultdict(int)
    for ms in machine_schedules:
        for ID, job_info in ms.job_lookup.items():
            s, d, job = job_info
            path, link = job.name
            scheduled_link_counts[link] += 1

    success = True
    for link, count in expected_link_counts.items():
        if count - scheduled_link_counts[link] != 0:
            print("Missing link {}!".format(link))
    return success


def generate_random_parameters(num_paths):
    # Number of nodes
    num_machines = 9

    # Decoherence weights for comm/storage qubits
    machine_params = [(0.68526781, 1.00004608) for _ in range(num_machines)]

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

    # Set of paths for building long distance entanglement
    paths = []
    for p in range(num_paths):
        # Select two end nodes
        start_node, end_node = sample(list(range(num_machines)), 2)
        if tuple(sorted([start_node, end_node])) in links:
            paths.append([start_node, end_node])

        else:
            Q = set(range(num_machines))
            dist = dict([(i, float('inf')) for i in range(num_machines)])
            prev = defaultdict()
            dist[start_node] = 0

            while Q:
                vertices = list(Q)
                distances = [dist[x] for x in vertices]
                u = vertices[distances.index(min(distances))]
                Q.remove(u)

                for l in links:
                    if u in l:
                        v = l[0] if u == l[1] else l[1]
                        if v in Q:
                            alt = dist[u] + 1
                            if alt < dist[v]:
                                dist[v] = alt
                                prev[v] = u

            u = end_node
            path = []
            while u is not None:
                path = [u] + path
                u = prev.get(u)

            paths.append(path)

    # (initial fidelity, expected generation time) for each link
    link_params = {}
    possible_gen_times = [0.1, 0.2, 0.3, 0.4, 0.5]
    for l in links:
        fidelity = randint(60, 100) / 100
        latency = choice(possible_gen_times)
        link_params[l] = (fidelity, latency)

    return num_machines, machine_params, links, link_params, paths


def main():
    network_params = demo_network_parameters()
    msbf, pfbf, ptbf = network_simulation(*network_params, brute_force_optimal_schedule)
    msh, pfh, pth = network_simulation(*network_params, heuristic_search_schedule)
    if [ms.get_job_view() for ms in msbf] == [ms.get_job_view() for ms in msh]:
        print("Brute force and Heuristic computed the same schedule!")
        status = verify_schedule(msbf, network_params)
        print("Schedule satisfies paths? {}".format(status))
        print("Results:")
        paths = network_params[-1]
        for path, fidelity, timespan in zip(paths, pfbf, ptbf):
            print("Path {}: Fidelity {}, Timespan {}".format(path, fidelity, timespan))
        print_schedules(msbf)


def main_random():
    num_tests = 10
    num_paths = 7
    test_output_file = "simulation_tests.txt"
    for test in range(num_tests):
        network_params = generate_random_parameters(num_paths)
        with open(test_output_file, 'a') as f:
            f.write("========================================================\n")
            f.write("Beginning test {}\n".format(test))
            f.write("Parameters:\n")
            f.write("Num machines: {}\nMachine Params: {}\nLinks: {}\nLink Params: {}\nPaths: {}\n".format(*network_params))

        start = time()
        msbf, pfbf, ptbf = network_simulation(*network_params, brute_force_optimal_schedule)
        end = time()
        tbf = end - start
        start = time()
        msbfa, pfbfa, ptbfa = network_simulation(*network_params, brute_force_adjacent)
        end = time()
        tbfa = end - start
        start = time()
        msh, pfh, pth = network_simulation(*network_params, heuristic_search_schedule)
        end = time()
        th = end - start
        start = time()
        mshd, pfhd, pthd = network_simulation(*network_params, heuristic_search_schedule_with_depth)
        end = time()
        thd = end - start
        with open(test_output_file, 'a') as f:
            f.write("Brute force took {}s\nBrute force adjacent took {}s\nHeuristic took {}s\nHeuristic with Depth took {}s\n".format(tbf, tbfa, th, thd))
        if [ms.get_job_view() for ms in msbf] == [ms.get_job_view() for ms in msbfa]:
            with open(test_output_file, 'a') as f:
                f.write("Brute force and Brute force adjacent computed the same schedule!\n")
        if [ms.get_job_view() for ms in msh] == [ms.get_job_view() for ms in mshd]:
            with open(test_output_file, 'a') as f:
                f.write("Heuristic and Heuristic with Depth computed the same schedule!\n")
        if [ms.get_job_view() for ms in msbf] == [ms.get_job_view() for ms in msh]:
            with open(test_output_file, 'a') as f:
                f.write("Brute force and Heuristic computed the same schedule!\n")
                status = verify_schedule(msbf, pfbf, network_params)
                f.write("Schedule satisfies paths? {}\n".format(status))
                f.write("Results:\n")
                paths = network_params[-1]
                for path, fidelity, timespan in zip(paths, pfbf, ptbf):
                    f.write("Path {}: Fidelity {}, Timespan {}\n".format(path, fidelity, timespan))
                f.write("==============================================================\n")
        elif [ms.get_job_view() for ms in msbf] == [ms.get_job_view() for ms in mshd]:
            with open(test_output_file, 'a') as f:
                f.write("Brute force and Heuristic with Depth computed the same schedule!\n")
                status = verify_schedule(msbf, pfbf, network_params)
                f.write("Schedule satisfies paths? {}\n".format(status))
                f.write("Results:\n")
                paths = network_params[-1]
                for path, fidelity, timespan in zip(paths, pfbf, ptbf):
                    f.write("Path {}: Fidelity {}, Timespan {}\n".format(path, fidelity, timespan))
                f.write("==============================================================\n")
        elif [ms.get_job_view() for ms in msbfa] == [ms.get_job_view() for ms in mshd]:
            with open(test_output_file, 'a') as f:
                f.write("Brute force adjacent and Heuristic with Depth computed the same schedule!\n")
                status = verify_schedule(msbfa, pfbfa, network_params)
                f.write("Schedule satisfies paths? {}\n".format(status))
                f.write("Results:\n")
                paths = network_params[-1]
                for path, fidelity, timespan in zip(paths, pfbfa, ptbfa):
                    f.write("Path {}: Fidelity {}, Timespan {}\n".format(path, fidelity, timespan))
                f.write("==============================================================\n")
        else:
            with open(test_output_file, 'a') as f:
                f.write("Brute force and Heuristics computed different schedules!\n")
                status_bf = verify_schedule(msbf, pfbf, network_params)
                status_h = verify_schedule(msh, pfh, network_params)
                f.write("BF Schedule satisfies paths? {}\n".format(status_bf))
                f.write("H Schedule satisfies paths? {}\n".format(status_h))
                f.write("Results BF:\n")
                paths = network_params[-1]
                for path, fidelity, timespan in zip(paths, pfbf, ptbf):
                    f.write("Path {}: Fidelity {}, Timespan {}\n".format(path, fidelity, timespan))
                f.write("Results H:\n")
                paths = network_params[-1]
                for path, fidelity, timespan in zip(paths, pfh, pth):
                    f.write("Path {}: Fidelity {}, Timespan {}\n".format(path, fidelity, timespan))
                f.write("==============================================================\n")


def test_specific_instance():
    num_machines = 9
    machine_params = [(0.68526781, 1.00004608), (0.68526781, 1.00004608), (0.68526781, 1.00004608), (0.68526781, 1.00004608),
                      (0.68526781, 1.00004608), (0.68526781, 1.00004608), (0.68526781, 1.00004608), (0.68526781, 1.00004608),
                      (0.68526781, 1.00004608)]
    links = [(0, 2), (1, 2), (2, 5), (2, 8), (3, 5), (4, 5), (5, 8), (6, 8), (7, 8)]
    link_params = {(0, 2): (0.99, 0.2), (1, 2): (0.81, 0.4), (2, 5): (0.78, 0.1), (2, 8): (0.7, 0.3), (3, 5): (0.86, 0.3), (4, 5): (0.67, 0.4), (5, 8): (0.65, 0.5), (6, 8): (0.96, 0.3), (7, 8): (1.0, 0.1)}
    paths = [[7, 8, 5, 3], [0, 2, 5, 4], [0, 2, 5, 3], [7, 8, 2, 1], [6, 8], [2, 5], [5, 2]]

    schedules = []
    # for i in range(1, len(paths)):
        # network_params = num_machines, machine_params, links, link_params, paths[:i]
    network_params = num_machines, machine_params, links, link_params, paths
    print("Running Brute Force")
    start = time()
    msbf, pfbf, ptbf = network_simulation(*network_params, brute_force_optimal_schedule)
    end = time()
    tbf = end - start
    print("Running Brute Force Adjacent")
    start = time()
    msbfa, pfbfa, ptbfa = network_simulation(*network_params, brute_force_adjacent)
    end = time()
    tbfa = end - start
    print("Running Heuristic")
    start = time()
    msh, pfh, pth = network_simulation(*network_params, heuristic_search_schedule)
    end = time()
    th = end - start
    print("Running Heuristic with Depth")
    start = time()
    mshd, pfhd, pthd = network_simulation(*network_params, heuristic_search_schedule_with_depth)
    end = time()
    thd = end - start
    schedules.append((msbf, msbfa, msh, mshd))
    pdb.set_trace()
    if [ms.get_job_view() for ms in msbf] == [ms.get_job_view() for ms in msbfa]:
        print("Brute force and Brute force adjacent computed the same schedule!")
    if [ms.get_job_view() for ms in msh] == [ms.get_job_view() for ms in mshd]:
        print("Heuristic and Heuristic with Depth computed the same schedule!")
    if [ms.get_job_view() for ms in msbf] == [ms.get_job_view() for ms in msh]:
        print("Brute force and Heuristic computed the same schedule!")
        status = verify_schedule(msbf, pfbf, network_params)
        print("Schedule satisfies paths? {}".format(status))
        print("Results:")
        paths = network_params[-1]
        for path, fidelity, timespan in zip(paths, pfbf, ptbf):
            print("Path {}: Fidelity {}, Timespan {}".format(path, fidelity, timespan))
        print_schedules(msbf)
    elif [ms.get_job_view() for ms in msbf] == [ms.get_job_view() for ms in mshd]:
        print("Brute force and Heuristic with Depth computed the same schedule!")
        status = verify_schedule(msbf, pfbf, network_params)
        print("Schedule satisfies paths? {}".format(status))
        print("Results:")
        paths = network_params[-1]
        for path, fidelity, timespan in zip(paths, pfbf, ptbf):
            print("Path {}: Fidelity {}, Timespan {}".format(path, fidelity, timespan))
        print_schedules(msbf)


if __name__ == '__main__':
    test_specific_instance()

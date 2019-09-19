import pdb
import numpy as np
from math import exp, log
from job import Job
from time import time

class LinkJob(Job):
    def __init__(self, name, ID, duration=0, params=None):
        super(LinkJob, self).__init__(name=name, ID=ID, duration=duration, params=params)

        self.w_left = None
        self.w_right = None
        self.loc_left = None
        self.loc_right = None
        self.F0 = None

    def set_dec_params(self, w_left, w_right):
        self.w_left = w_left
        self.w_right = w_right

    def get_dec_params(self):
        # if self.w_left is None or self.w_right is None:
        #     raise Exception("Job does not have left and right node decoherence weights assigned!")

        return self.w_left, self.w_right

    def set_loc_info(self, loc_left, loc_right):
        self.loc_left = loc_left
        self.loc_right = loc_right

    def get_loc_info(self):
        if self.loc_left is None or self.loc_right is None:
            raise Exception("Job does not have left and right node storage locations assigned!")

        return self.loc_left, self.loc_right

    def set_initial_fidelity(self, F0):
        self.F0 = F0

    def get_initial_fidelity(self):
        if self.F0 is None:
            raise Exception("No initial fidelity for link!")

        return self.F0


def compute_swapped_fidelity(Fx, Fy):
    Fswap = Fx * Fy + (1 - Fx) * (1 - Fy) / 3
    return Fswap


def compute_swapped_fidelity_fit(Fx, Fy):
    Fswap = 0.81356455 * Fx * Fy + -0.11015269*(1 - Fx) * (1 - Fy) / 3 + 0.05616237
    return Fswap


def three_link_swap_fidelity(Fx, Fy, Fz):
    Fxy = Fx*Fy
    Fxz = Fx*Fz
    Fyz = Fy*Fz
    Fxyz = Fxy*Fz
    return (2 + (Fx + Fy + Fz) - 4*(Fxy + Fyz + Fxz) + 16 * Fxyz) / 9


def three_link_swap_fidelity_fit(Fx, Fy, Fz):
    return compute_swapped_fidelity_fit(compute_swapped_fidelity_fit(Fx, Fy), Fz)


def decohered_fidelity(F0, w, t):
    try:
        t0 = -log(2 * F0 - 1) / w
        return (exp(-w * (t0 + t)) + 1) / 2
    except:
        return 0.5


def apply_job_move_noise(F, job):
    if job.loc_left == "stor":
        F = apply_move_noise(F, 0.94968872, 0.01672558)
    if job.loc_right == "stor":
        F = apply_move_noise(F, 0.94968872, 0.01672558)
    return F


def apply_move_noise(F, a, b):
    return a*F + b


def assign_qubits(machine_params, jobs, end_times):
    jobs[0].w_left = machine_params[0][0]
    jobs[0].loc_left = "comm"
    jobs[-1].w_right = machine_params[-1][0]
    jobs[-1].loc_right = "comm"

    for i, info in enumerate(zip(zip(jobs[:], jobs[1:]), zip(end_times[:], end_times[1:]))):
        job_pair, end_pair = info
        jl, jr = job_pair
        el, er = end_pair

        if el < er:
            jl.w_right = machine_params[i+1][0]
            jl.loc_right = "comm"
            jr.w_left = machine_params[i+1][1]
            jr.loc_left = "stor"
        else:
            jl.w_right = machine_params[i + 1][1]
            jl.loc_right = "stor"
            jr.w_left = machine_params[i + 1][0]
            jr.loc_left = "comm"

    return jobs


def compute_segment_fidelity(jobs, job_times, time_swap):
    if len(jobs) == 0:
        return 1

    # Final fidelity of a segment of a single link is just the decohered link fidelity
    if len(jobs) == 1:
        [job] = jobs
        [gen_time] = job_times
        w = sum(job.get_dec_params())
        F0 = job.get_initial_fidelity()
        F0 = apply_job_move_noise(F0, job)
        return decohered_fidelity(F0, w, t=time_swap - gen_time)

    # If there are two links, compute the fidelity of the earlier link, decohere, swap, and decohere
    elif len(jobs) == 2:
        job_left, job_right = jobs
        gen_left, gen_right = job_times
        wl, _ = job_left.get_dec_params()
        _, wr = job_right.get_dec_params()
        wswap = wl + wr
        if gen_left >= gen_right:
            w = sum(job_right.get_dec_params())
            Fr = job_right.get_initial_fidelity()
            Fr = apply_job_move_noise(Fr, job_right)
            Fr = decohered_fidelity(Fr, w, t=gen_left-gen_right)
            Fl = job_left.get_initial_fidelity()
            Fl = apply_job_move_noise(Fl, job_left)
            Fswap = compute_swapped_fidelity(Fr, Fl)
            Fswap_decohered = decohered_fidelity(Fswap, wswap, time_swap - gen_left)

        else:
            w = sum(job_left.get_dec_params())
            Fl = job_left.get_initial_fidelity()
            Fl = apply_job_move_noise(Fl, job_left)
            Fl = decohered_fidelity(Fl, w, t=gen_right - gen_left)
            Fr = job_right.get_initial_fidelity()
            Fr = apply_job_move_noise(Fr, job_right)
            Fswap = compute_swapped_fidelity(Fl, Fr)
            Fswap_decohered = decohered_fidelity(Fswap, wswap, time_swap - gen_right)

        return Fswap_decohered

    else:
        last_job_index = job_times.index(max(job_times))
        jobs_left = jobs[:last_job_index]
        jobs_right = jobs[last_job_index + 1:]
        if jobs_left and not jobs_right:
            Fl = compute_segment_fidelity(jobs_left, job_times[:last_job_index], time_swap=job_times[last_job_index])
            last_job = jobs[last_job_index]
            Fm = last_job.get_initial_fidelity()
            Fm = apply_job_move_noise(Fm, last_job)
            return compute_swapped_fidelity(Fl, Fm)
        elif jobs_right and not jobs_left:
            Fr = compute_segment_fidelity(jobs_right, job_times[last_job_index+1:], time_swap=job_times[last_job_index])
            last_job = jobs[last_job_index]
            Fm = last_job.get_initial_fidelity()
            Fm = apply_job_move_noise(Fm, last_job)
            return compute_swapped_fidelity(Fr, Fm)
        else:
            Fl = compute_segment_fidelity(jobs_left, job_times[:last_job_index], time_swap=job_times[last_job_index])
            Fr = compute_segment_fidelity(jobs_right, job_times[last_job_index+1:], time_swap=job_times[last_job_index])
            last_job = jobs[last_job_index]
            Fm = last_job.get_initial_fidelity()
            Fm = apply_job_move_noise(Fm, last_job)
            return three_link_swap_fidelity(Fl, Fm, Fr)


def compute_full_schedule_fidelity(jobs, slot_times):
    time_last_swap = max(slot_times)
    return compute_segment_fidelity(jobs, slot_times, time_last_swap)


def brute_force_optimal_schedule(machine_params, jobs, available_slots):
    selected_slots = []
    F, T, slots = brute_force_helper(machine_params, jobs, available_slots, selected_slots, 0)
    return F, T, slots


def remove_overlapping_slots(slot, slots):
    return list(filter(lambda x: not (slot[0] <= x[0] <= slot[1] or slot[0] <= x[1] <= slot[1]), slots))


def remove_nonadjacent_slots(slot, slots):
    if slot[0] > slots[-1][1]:
        return [slots[-1]]
    elif slot[1] < slots[0][0]:
        return [slots[0]]
    else:
        slot_pre = None
        slot_post = None
        for possible_slot in slots:
            s, e = slot
            ps, pe = possible_slot
            if s > pe:
                slot_pre = possible_slot
            elif e < ps and slot_post is None:
                slot_post = possible_slot
        adjacent_slots = []
        if slot_pre is not None:
            adjacent_slots.append(slot_pre)
        if slot_post is not None:
            adjacent_slots.append(slot_post)
        return adjacent_slots


def brute_force_helper(machine_params, jobs, available_slots, selected_slots, i):
    if i == len(jobs) - 1:
        bestF = 0
        bestT = float('inf')
        best_assignment = []
        for slot in available_slots[i]:
            possible_assignment = selected_slots + [slot]
            possible_end_times = [s[1] for s in possible_assignment]
            updated_jobs = assign_qubits(machine_params, jobs, possible_end_times)
            F = compute_full_schedule_fidelity(updated_jobs, possible_end_times)
            T = np.round(max([s[1] for s in possible_assignment]) - min([s[0] for s in possible_assignment]), decimals=1)
            if bestF == 0 or (F > bestF and T <= bestT) or (F >= bestF and T < bestT):
                bestF = F
                bestT = T
                best_assignment = possible_assignment

        return bestF, bestT, best_assignment

    bestF = 0
    bestT = float('inf')
    best_assignment = []
    for slot in available_slots[i]:
        # Remove the current slot we're looking at from the next schedules availability since we occupy the node
        next_node_slots = remove_overlapping_slots(slot, available_slots[i+1])
        altered_available_slots = available_slots[:i+1] + [next_node_slots] + available_slots[i+2:]
        altered_selected_slots = selected_slots + [slot]
        F, T, assignment = brute_force_helper(machine_params, jobs, altered_available_slots, altered_selected_slots, i + 1)
        if bestF == 0 or (F > bestF and T <= bestT):
            bestF = F
            bestT = T
            best_assignment = assignment

    return bestF, bestT, best_assignment


def heuristic_search_schedule(machine_params, jobs, available_slots):
    best_slots = []
    bestF = 0
    bestT = float('inf')
    for slot in available_slots[0]:
        selected_slots = [slot]
        for i in range(1, len(jobs)):
            adjacent_slots = remove_nonadjacent_slots(selected_slots[-1], available_slots[i])
            bestCurrF = 0
            bestCurrT = float('inf')
            bestSlot = None
            for aslot in adjacent_slots:
                possible_assignment = selected_slots + [aslot]
                possible_end_times = [s[1] for s in possible_assignment]
                assign_qubits(machine_params[:i+1], jobs[:i+1], possible_end_times)
                F = compute_full_schedule_fidelity(jobs[:i+1], possible_end_times)
                T = np.round(max([s[1] for s in selected_slots + [aslot]]) - min([s[0] for s in possible_assignment]), decimals=1)
                if bestCurrF == 0 or (F > bestCurrF and T <= bestCurrT):
                    bestCurrF = F
                    bestCurrT = T
                    bestSlot = aslot

            selected_slots.append(bestSlot)

        possible_end_times = [s[1] for s in selected_slots]
        assign_qubits(machine_params, jobs, possible_end_times)
        F = compute_full_schedule_fidelity(jobs, possible_end_times)
        T = np.round(max([s[1] for s in selected_slots]) - min([s[0] for s in selected_slots]), decimals=1)
        if bestF == 0 or (F > bestF and T <= bestT):
            bestF = F
            bestT = T
            best_slots = selected_slots

    return bestF, bestT, best_slots


def main():
    # Pairs of decoherence weights on the comm/storage q locations
    num_machines = 20

    # Decoherence weights of the (comm_q, storage_q)
    machine_params = [(0.01, 0.01) for _ in range(num_machines)]

    # (initial fidelity, expected generation time) for each link
    link_params = [(0.9, 1) for _ in range(num_machines - 1)]

    # Available schedule slots (assume each link takes 1 time unit)
    time_unit = 0.1  # 100ms
    available_slots = [[(j*time_unit, time_unit*(j + link_params[i][1] - 1)) for j in range(5)] for i in range(num_machines - 1)]

    # Create job structures
    jobs = []
    for i, info in enumerate(zip(link_params, zip(machine_params[:], machine_params[1:]))):
        lp, mps = info
        lj = LinkJob(name=i, ID=i)
        initial_fidelity, _ = lp
        lj.set_initial_fidelity(F0=initial_fidelity)
        jobs.append(lj)

    start_time = time()
    fidelity, timespan, slots = brute_force_optimal_schedule(machine_params, jobs, available_slots)
    fin_time = time()
    print("Brute force found schedule with fidelity {} and timespan {} using slot assignment {}".format(fidelity, timespan, slots))
    print("Took {}s".format(fin_time - start_time))
    start_time = time()
    fidelity, timespan, slots = heuristic_search_schedule(machine_params, jobs, available_slots)
    fin_time = time()
    print("Heuristic found schedule with fidelity {} and timespan {} using slot assignment {}".format(fidelity, timespan, slots))
    print("Took {}s".format(fin_time - start_time))
    pdb.set_trace()


if __name__ == "__main__":
    main()

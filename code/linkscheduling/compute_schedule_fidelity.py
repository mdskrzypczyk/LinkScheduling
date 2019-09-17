import pdb
from math import exp, log
from job import Job


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


def three_link_swap_fidelity(Fx, Fy, Fz):
    Fxy = Fx*Fy
    Fxz = Fx*Fz
    Fyz = Fy*Fz
    Fxyz = Fxy*Fz
    return (2 + (Fx + Fy + Fz) - 4*(Fxy + Fyz + Fxz) + 16 * Fxyz) / 9


def decohered_fidelity(F0, w, t):
    t0 = -log(2 * F0 - 1) / w
    return (exp(-w * (t0 + t)) + 1) / 2


def compute_segment_fidelity(jobs, job_times, time_swap):
    if len(jobs) == 0:
        return 1

    # Final fidelity of a segment of a single link is just the decohered link fidelity
    if len(jobs) == 1:
        [job] = jobs
        [gen_time] = job_times
        w = sum(job.get_dec_params())
        F0 = job.get_initial_fidelity()
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
            Fr = decohered_fidelity(Fr, w, t=gen_left-gen_right)
            Fswap = compute_swapped_fidelity(Fr, job_left.get_initial_fidelity())
            Fswap_decohered = decohered_fidelity(Fswap, wswap, time_swap - gen_left)

        else:
            w = sum(job_left.get_dec_params())
            Fl = job_left.get_initial_fidelity()
            Fl = decohered_fidelity(Fl, w, t=gen_right - gen_left)
            Fswap = compute_swapped_fidelity(Fl, job_right.get_initial_fidelity())
            Fswap_decohered = decohered_fidelity(Fswap, wswap, time_swap - gen_right)

        return Fswap_decohered

    else:
        last_job_index = job_times.index(max(job_times))
        jobs_left = jobs[:last_job_index]
        jobs_right = jobs[last_job_index + 1:]
        Fl = compute_segment_fidelity(jobs_left, job_times[:last_job_index], time_swap=job_times[last_job_index])
        Fr = compute_segment_fidelity(jobs_right, job_times[last_job_index+1:], time_swap=job_times[last_job_index])
        last_job = jobs[last_job_index]
        return three_link_swap_fidelity(Fl, last_job.get_initial_fidelity(), Fr)


def compute_full_schedule_fidelity(jobs, slot_times):
    time_last_swap = max(slot_times)
    return compute_segment_fidelity(jobs, slot_times, time_last_swap)


def brute_force_optimal_schedule(machine_params, jobs, available_slots):
    selected_slots = []
    F, slots = brute_force_helper(machine_params, jobs, available_slots, selected_slots, 0)
    return F, slots


def brute_force_helper(machine_params, jobs, available_slots, selected_slots, i):
    if i == len(jobs) - 1:
        bestF = 0
        best_assignment = []
        for slot in available_slots[i]:
            possible_assignment = selected_slots + [slot]
            possible_end_times = [s[1] for s in possible_assignment]
            updated_jobs = assign_qubits(machine_params, jobs, possible_end_times)
            F = compute_full_schedule_fidelity(updated_jobs, possible_end_times)
            if bestF == 0 or F > bestF:
                bestF = F
                best_assignment = possible_assignment

        return bestF, best_assignment

    bestF = 0
    best_assignment = []
    for slot in available_slots[i]:
        # Remove the current slot we're looking at from the next schedules availability since we occupy the node
        next_node_slots = list(filter(lambda x: not (slot[0] <= x[0] <= slot[1] or slot[0] <= x[1] <= slot[1]), available_slots[i + 1]))
        altered_available_slots = available_slots[:i+1] + [next_node_slots] + available_slots[i+2:]
        altered_selected_slots = selected_slots + [slot]
        F, assignment = brute_force_helper(machine_params, jobs, altered_available_slots, altered_selected_slots, i + 1)
        if bestF == 0 or F > bestF:
            bestF = F
            best_assignment = assignment

    return bestF, best_assignment


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


def main():
    # Pairs of decoherence weights on the comm/storage q locations
    num_machines = 6

    # Decoherence weights of the (comm_q, storage_q)
    machine_params = [(0.01, 0.01) for _ in range(num_machines)]

    # (initial fidelity, expected generation time) for each link
    link_params = [(0.9, 2) for _ in range(num_machines - 1)]

    # Available schedule slots (assume each link takes 1 time unit)
    time_unit = 1000000  # 1ms
    available_slots = [[(j*time_unit, time_unit*(j + link_params[i][1] - 1)) for j in range(5)] for i in range(num_machines - 1)]

    # Create job structures
    jobs = []
    for i, info in enumerate(zip(link_params, zip(machine_params[:], machine_params[1:]))):
        lp, mps = info
        lj = LinkJob(name=i, ID=i)
        initial_fidelity, _ = lp
        lj.set_initial_fidelity(F0=initial_fidelity)
        jobs.append(lj)

    fidelity, slots = brute_force_optimal_schedule(machine_params, jobs, available_slots)
    print("Found schedule with fidelity {} using slot assignment {}".format(fidelity, slots))
    pdb.set_trace()


if __name__ == "__main__":
    main()

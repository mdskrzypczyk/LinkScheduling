import copy
from job import Job, JobOrdering
from schedule import LinkSchedule
from random import randint
from math import ceil

# Assume all schedules are clear so links can be scheduled immediately after/before one another
# Assume each node has a single communication qubit (should be extendable if a schedule is provided for each of
# of the communication qubits and one is selected)
# Dynamic programming to brute force optimal solutions


def compute_swapped_fidelity(f1, f2):
    fswap = f1*f2 + (1-f1)*(1-f2)/3
    return fswap


def greedy(jobs, schedules):
    pass


def compare_orderings(jobs, ordering1, ordering2):
    # Select ordering with lower total duration
    # return compute_duration(jobs, ordering1) < compute_duration(jobs, ordering2)

    # Select ordering with lower idle time
    return compute_idle_time(jobs, ordering1) < compute_idle_time(jobs, ordering2)


def compute_idle_time(jobs, jobordering):
    rel_start = 0
    abs_start = rel_start
    rel_completion = jobs[0].duration
    abs_completion = rel_completion

    jobs[0].set_start(0)
    for job, placement in zip(jobs[1:], jobordering.ordering[1:]):
        if placement is True:
            rel_start = rel_completion
            job.set_start(rel_start)
            rel_completion = rel_start + job.duration
            if rel_completion > abs_completion:
                abs_completion = rel_completion
        elif placement is False:
            rel_completion = rel_start
            rel_start = rel_completion - job.duration
            job.set_start(rel_start)
            if rel_start < abs_start:
                abs_start = rel_start

    for job in jobs:
        job.start -= abs_start

    abs_completion -= abs_start

    idle_time = 0
    for i in range(len(jobs) - 1):
        idle_time += abs((jobs[i].start + jobs[i].duration) - (jobs[i+1].start + jobs[i+1].duration))
    idle_time += abs_completion - (jobs[0].start + jobs[0].duration)
    idle_time += abs_completion - (jobs[-1].start + jobs[-1].duration)
    return idle_time


def compute_duration(jobs, jobordering):
    rel_start = 0
    abs_start = rel_start
    rel_completion = jobs[0].duration
    abs_completion = rel_completion
    for job, placement in zip(jobs[1:], jobordering.ordering[1:]):
        if placement is True:
            rel_start = rel_completion
            rel_completion = rel_start + job.duration
            if rel_completion > abs_completion:
                abs_completion = rel_completion
        elif placement is False:
            rel_completion = rel_start
            rel_start = rel_completion - job.duration
            if rel_start < abs_start:
                abs_start = rel_start

    return abs_completion - abs_start


def compute_link_availability(schedules):
    link_schedules = []
    for schedule_left, schedule_right in zip(schedules[:], schedules[1:]):
        left_availability = schedule_left.get_availability_view()
        right_availability = schedule_right.get_availability_view()
        combined_availability = [l & r for l, r in zip(left_availability, right_availability)]
        ls = LinkSchedule(availability_view=combined_availability)
        link_schedules.append(ls)

    return link_schedules


def brute_force_naive(jobs, schedules):
    job_ordering = JobOrdering(size=len(jobs))
    job_ordering.set_rel_start_post(0)
    job_ordering = brute_force_helper(jobs, job_ordering, index=1)
    return job_ordering


def brute_force_helper(jobs, job_ordering, index):
    job_ordering_pre = copy.copy(job_ordering)
    job_ordering_pre.set_rel_start_pre(index)
    job_ordering_post = job_ordering
    job_ordering_post.set_rel_start_post(index)

    if index < len(jobs) - 1:
        job_ordering_pre = brute_force_helper(jobs, job_ordering_pre, index + 1)
        job_ordering_post = brute_force_helper(jobs, job_ordering_post, index + 1)

    if compare_orderings(jobs, job_ordering_pre, job_ordering_post):
        return job_ordering_pre
    else:
        return job_ordering_post


def compute_job_ordering(jobs, link_schedules):
    # First check if each link schedule has an available slot for it's job
    potential_job_slots = []
    for j, ls in zip(jobs, link_schedules):
        num_slots_for_job = ceil(j.duration / ls.get_slot_size())
        c_slots = ls.get_continuous_slots(min_size=num_slots_for_job)
        if not c_slots:
            return False
        potential_job_slots.append(c_slots)

    # Start with the most constrained link
    slot_counts = [len(c_slots) for c_slots in potential_job_slots]
    link_index = slot_counts.index(min(slot_counts))

    # Iterate over the possible starting indexes and recursively build a job ordering
    slot_assignment_metrics = []
    potential_slots = []
    for c_slot in potential_job_slots[link_index]:
        initial_slots = [None for _ in range(link_index)] + [c_slot] + [None for _ in range(link_index+1,len(jobs))]
        metric, used_slots = compute_remaining_jobs(link_index, initial_slots, jobs, potential_job_slots)
        slot_assignment_metrics.append(metric)
        potential_slots.append(used_slots)

    # Take the best assignment
    best_index = slot_assignment_metrics.index(max(slot_assignment_metrics))
    used_slots = potential_slots[best_index]
    return slot_assignment_metrics[best_index], used_slots


def compute_remaining_jobs(last_placed_index, current_slots, jobs, potential_job_slots):
    # Use a window to prune continuous slots that for adjacent links that would be generated too far away
    last_used_slots = current_slots[last_placed_index]
    last_start, last_end = last_used_slots
    potential_job_slots_copy = list(potential_job_slots)
    if last_placed_index > 0 and current_slots[last_placed_index-1] != None:
        # Compute the window???
        tolerance_window_left_pre = xxx
        tolerance_window_left_post = xxx
        pruned_job_slots_left = []
        for c_slot in potential_job_slots[last_placed_index-1]:
            c_start, c_end = c_slot
            if last_start - tolerance_window_left_pre <= c_end:
                pruned_job_slots_left.append(c_slot)
            elif last_end + tolerance_window_left_post >= c_end:
                pruned_job_slots_left.append(c_slot)

        potential_job_slots_copy[last_placed_index-1] = pruned_job_slots_left

    if last_placed_index < len(jobs) - 1 and current_slots[last_placed_index+1] != None:
        # Compute the window???
        tolerance_window_right_pre = xxx
        tolerance_window_right_post = xxx
        pruned_job_slots_right = []
        for c_slot in potential_job_slots[last_placed_index + 1]:
            c_start, c_end = c_slot
            if last_start - tolerance_window_right_pre <= c_end:
                pruned_job_slots_right.append(c_slot)
            elif last_end + tolerance_window_right_post >= c_end:
                pruned_job_slots_right.append(c_slot)

        potential_job_slots_copy[last_placed_index + 1] = pruned_job_slots_right

    for left_slot in potential_job_slots_copy[last_placed_index - 1]:
        for right_slot in potential_job_slots_copy[last_placed_index + 1]:
            pass



def insert_jobs_into_schedules(jobs, slots, schedules):
    clock = schedules[0].clock
    curr_time = clock.get_time()
    for i, js in enumerate(zip(jobs, slots)):
        job, slots = js
        slot_start, slot_end = js
        start_time = curr_time + slot_start * schedule_left.get_slot_size()
        schedule_left = schedules[i]
        schedule_right = schedules[i+1]
        schedule_left.insert_job(job=job, start_time=start_time, duration=job.duration)
        schedule_right.insert_job(job=job, start_time=start_time, duration=job.duration)


def compute_schedule(jobs, schedules):
    link_schedules = compute_link_availability(schedules)
    metric, slots = compute_job_ordering(jobs, link_schedules)
    insert_jobs_into_schedules(jobs, slots, schedules)


def main():
    job_names = "ABCDEFGH"
    weights = [randint(1, 10) for n in job_names]
    jobs = [Job(name=c, duration=w) for w, c in zip(weights, job_names)]
    schedules = None
    job_ordering = brute_force(jobs, schedules)
    print(weights)
    print(job_ordering.ordering)


if __name__ == '__main__':
    main()

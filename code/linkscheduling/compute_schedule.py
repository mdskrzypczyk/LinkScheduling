import copy
from job import Job, JobOrdering
from random import randint

# Assume all schedules are clear so links can be scheduled immediately after/before one another
# Assume each node has a single communication qubit (should be extendable if a schedule is provided for each of
# of the communication qubits and one is selected)
# Dynamic programming to brute force optimal solutions


def brute_force(jobs, schedules):
    job_ordering = JobOrdering(size=len(jobs))
    job_ordering.set_rel_start_post(0)
    job_ordering = brute_force_helper(jobs, job_ordering, index=1)
    print(compute_idle_time(jobs, job_ordering))
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


def greedy(jobs, schedules):



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


def main():
    job_names = "ABCDEFGH"
    weights = [randint(1, 10) for n in job_names]
    jobs = [Job(name=c, duration=w) for w, c in zip(weights, job_names)]
    schedules = None
    job_ordering = brute_force(jobs, schedules)
    print(weights)
    print(job_ordering.ordering)

if __name__=='__main__':
    main()

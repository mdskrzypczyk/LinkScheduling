import numpy as np
from collections import defaultdict


def get_wcrt_in_slots(schedule, slot_size):
    task_wcrts = defaultdict(int)
    for s, e, t in schedule:
        name_components = t.name.split("|")
        instance_name = "|".join(name_components[:2])
        task_wcrts[instance_name] = (e - t.a)*slot_size

    original_task_wcrts = defaultdict(int)
    for instance_name, wcrt in task_wcrts.items():
        original_taskname = instance_name.split("|")[0]
        original_task_wcrts[original_taskname] = max(original_task_wcrts[original_taskname], wcrt)

    return original_task_wcrts


def get_start_jitter_in_slots(taskset, schedule, slot_size):
    periodic_task_starts = defaultdict(list)
    for s, e, t in schedule:
        name_components = t.name.split("|")
        original_taskname = name_components[0]
        if len(name_components) == 3:
            if name_components[2] == "0":
                periodic_task_starts[original_taskname].append(s*slot_size)
        else:
            periodic_task_starts[original_taskname].append(s*slot_size)

    periodic_task_start_jitter = defaultdict(int)
    for periodic_task in taskset:
        task_starts = periodic_task_starts[periodic_task.name]
        change = []
        for s1, s2 in zip(task_starts, task_starts[1:]):
            diff = s2 - s1 - periodic_task.p * slot_size
            change.append(diff)

        if change:
            jitter = np.var(change)
            periodic_task_start_jitter[periodic_task.name] = jitter

    return periodic_task_start_jitter
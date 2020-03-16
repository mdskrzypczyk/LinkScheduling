import networkx as nx
from jobscheduling.schedulers.scheduler import get_lcm_for
from math import ceil
from networkx.algorithms.flow import maximum_flow
from collections import defaultdict
from jobscheduling.task import ResourceTask
from jobscheduling.visualize import schedule_and_resource_timelines
from jobscheduling.schedulers.scheduler import verify_schedule


class UniResourceFlowScheduler:
    def schedule_tasks(self, periodic_taskset, topology):
        schedule = networkflowschedule(periodic_taskset)
        return [(periodic_taskset, schedule, schedule is not None)]

def networkflowschedule(taskset):
    G = nx.DiGraph()
    hyperperiod = get_lcm_for([t.p for t in taskset])
    f_min = max([t.c for t in taskset])
    periods = list(sorted(set([t.p for t in taskset])))
    for period in periods:
        if period >= f_min:
            f_length = period
            break

    num_frame_nodes = hyperperiod // f_length
    G.add_node("source")
    G.add_node("sink")
    for i in range(num_frame_nodes):
        frame_name = "frame-{}".format(i)
        G.add_node(frame_name)
        G.add_edge(frame_name, "sink", capacity=f_length)

    for periodic_task in taskset:
        for instance in range(hyperperiod // periodic_task.p):
            node_name = "{}|{}".format(periodic_task.name, instance)
            task_instance = ResourceTask(name=node_name, a=instance * periodic_task.p + periodic_task.a, c=periodic_task.c,
                                         d=(instance + 1) * periodic_task.p + periodic_task.a, resources=periodic_task.resources)

            G.add_node(node_name, task_instance=task_instance)
            G.add_edge("source", node_name, capacity=periodic_task.c)
            f_initial = ceil((instance * periodic_task.p + periodic_task.a) / f_length)
            f_final = ceil(((instance + 1) * periodic_task.p + periodic_task.a) / f_length)
            for fid in range(f_initial, f_final + 1):
                G.add_edge(node_name, "frame-{}".format(fid), capacity=periodic_task.c)

    result = maximum_flow(G, "source", "sink", )
    frame_schedules = extract_frame_schedules(taskset, hyperperiod, G, result, f_length)
    if not frame_schedules:
        return None

    schedule = []
    for i in range(num_frame_nodes):
        frame_name = "frame-{}".format(i)
        schedule += frame_schedules[frame_name]

    new_schedule = []
    completed_instances = defaultdict(int)
    for i in range(num_frame_nodes):
        frame_name = "frame-{}".format(i)
        frame_schedule = frame_schedules[frame_name]
        for entry in sorted(frame_schedule, key=lambda entry: entry[2].d):
            start, end, task = entry
            original_taskname, instance = task.name.split('|')
            instance = int(instance)
            if completed_instances[original_taskname] == instance:
                if new_schedule:
                    start = new_schedule[-1][1]
                    start = max(start, task.a)
                    new_schedule.append((start, start+task.c, task))
                else:
                    new_schedule.append((0, task.c, task))
                completed_instances[original_taskname] += 1

    if verify_schedule(taskset, new_schedule):
        return new_schedule

    return None


def extract_frame_schedules(taskset, hyperperiod, G, result, f_length):
    frame_schedules = {}
    for periodic_task in taskset:
        for instance in range(hyperperiod // periodic_task.p):
            node_name = "{}|{}".format(periodic_task.name, instance)
            edge_flows = result[1][node_name]
            if sum(edge_flows.values()) != periodic_task.c:
                return None

            for edge, flow in edge_flows.items():
                if flow != 0:
                    task_instance = G.nodes[node_name]["task_instance"]
                    if not frame_schedules.get(edge):
                        frame_id = int(edge.split("-")[1])
                        frame_schedules[edge] = [(frame_id * f_length, frame_id * f_length + flow, task_instance)]
                    else:
                        start = frame_schedules[edge][-1][1]
                        frame_schedules[edge].append((start, start + flow, task_instance))

    return frame_schedules
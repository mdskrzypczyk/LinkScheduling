from itertools import product
from mip import Model, xsum, BINARY
from jobscheduling.schedulers.scheduler import get_lcm_for, verify_schedule
from jobscheduling.task import ResourceTask


class MultipleResourceILPBlockNPEDFScheduler:
    def schedule_tasks(self, taskset, topology):
        return ilpschedule(taskset, topology)


def get_resource_string(resource):
    resource_node, resource_id = resource.split('-')
    resource_type = resource_id[0]
    return resource_node + resource_type


def ilpschedule(taskset, topology):
    nodeG, G = topology

    # Resource capacities
    c = []
    resource_to_id = {}
    rid = 0
    for nodeID in sorted(G.nodes.keys()):
        node = G.nodes[nodeID]
        comm_resources = node["comm_qs"]
        storage_resources = node["storage_qs"]
        cid = get_resource_string(comm_resources[0])
        resource_to_id[cid] = rid
        c.append(len(comm_resources))
        rid += 1
        if storage_resources:
            sid = get_resource_string(storage_resources[0])
            resource_to_id[sid] = rid
            c.append(len(storage_resources))
            rid += 1

    # Processing times and resource utilization
    p = []
    u = []
    a = []
    d = []
    hyperperiod = get_lcm_for([t.p for t in taskset])
    jobid_to_instance = {}
    for periodic_task in taskset:
        num_instances = hyperperiod // periodic_task.p
        task_u = []
        resources = list(sorted([get_resource_string(r) for r in periodic_task.resources]))
        for r in sorted(resource_to_id.keys()):
            task_u.append(resources.count(r))

        for instance in range(num_instances):
            a.append(instance * periodic_task.p)
            d.append((instance + 1) * periodic_task.p)
            p.append(periodic_task.c)
            u.append(list(task_u))
            task_instance = ResourceTask(name="{}|{}".format(periodic_task.name, instance), c=periodic_task.c, a=a[-1],
                                         d=d[-1], resources=periodic_task.resources)
            jobid_to_instance[len(a) - 1] = task_instance

    (R, J, T) = (range(len(c)) , range(len(p)) , range(hyperperiod))

    model = Model()

    x = [[model.add_var(name="x({},{})".format(j, t), var_type=BINARY) for t in T] for j in J]

    model.objective = xsum(0 for t in T)

    for j in J:
        model += xsum(x[j][t] for t in T) == 1

    for (r, t) in product(R, T):
        model += (
                xsum(u[j][r] * x[j][t2] for j in J for t2 in range(max(0, t - p[j] + 1), t + 1))
                <= c[r])

    for j in J:
        model += xsum(t * x[j][t] for t in T) >= a[j]
        model += xsum(t * x[j][t] for t in T) <= d[j] - p[j]

    model.optimize()

    schedule = []
    for (j, t) in product(J, T):
        if x[j][t].x >= 0.99:
            task_instance = jobid_to_instance[j]
            schedule.append((t, t + task_instance.c, task_instance))

    return [(taskset, schedule, verify_schedule(taskset, schedule))]
from collections import defaultdict

class BlockResourceEDFScheduler:
    def __init__(self):
        pass

    def schedule(self, tasks, resources):
        for t in tasks:
            priority = t.D
            for resource in t.resources:
                resource.add_to_queue((priority, t))

        while tasks:
            ready_tasks = []
            resource_count = defaultdict(int)
            for t in tasks:
                for resource in t.resources:
                    p, t = resource.get_head()
                    resource_count[t.name] += 1

            for t in tasks:
                if resource_count[t.name] == t.num_resources:
                    ready_tasks.append(t)
                    for resource in t.resources:
                        resource.pop_from_queue()
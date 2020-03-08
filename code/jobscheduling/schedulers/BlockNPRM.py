import networkx as nx
from copy import copy
from collections import defaultdict
from intervaltree import Interval, IntervalTree
from queue import PriorityQueue
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler
from jobscheduling.task import get_lcm_for, ResourceTask, PeriodicResourceDAGTask


logger = LSLogger()


class UniResourceBlockNPRMScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p*instance
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = ResourceTask(name="{}|{}".format(dag_copy.name, instance), c=dag_copy.c, a=release_offset,
                                    d=dag_copy.a + dag_copy.p*(instance + 1), resources=dag_copy.resources)
        return dag_instance

    def initialize_taskset(self, tasks):
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    def schedule_tasks(self, taskset, topology):
        original_taskset = taskset
        taskset = self.preprocess_taskset(taskset)
        queue = PriorityQueue()
        schedule = []

        # First sort the taskset by activation time
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])
        taskset = self.initialize_taskset(taskset)
        taskset = sorted(taskset, key=lambda task: (task.a, taskset_lookup[task.name.split('|')[0]].p))

        # Let time evolve and simulate scheduling, start at first task
        curr_time = taskset[0].a
        while taskset or not queue.empty():
            while taskset and taskset[0].a <= curr_time:
                task = taskset.pop(0)
                queue.put((task.d, task))

            if not queue.empty():
                priority, next_task = queue.get()
                schedule.append((curr_time, curr_time + next_task.c, next_task))
                original_taskname, instance = next_task.name.split('|')
                instance = int(instance)
                if instance < instance_count[original_taskname]:
                    periodic_task = taskset_lookup[original_taskname]
                    task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                    taskset = list(sorted(taskset + [task_instance], key=lambda task: (task.a, taskset_lookup[task.name.split('|')[0]].p)))

                curr_time += next_task.c

            elif taskset:
                curr_time = taskset[0].a

        # Check validity
        valid = True
        for start, end, task in schedule:
            if task.d < end:
                valid = False
        taskset = original_taskset

        return [(taskset, schedule, valid)]

    def check_feasibility(self, taskset):
        pass



class MultiResourceBlockNPRMScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def initialize_taskset(self, tasks):
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p*instance
        task_instance = ResourceTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset, c=dag_copy.c,
                                     d=dag_copy.a + dag_copy.p*(instance + 1), resources=dag_copy.resources)
        return task_instance

    def schedule_tasks(self, taskset, topology):
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = dict([(t.name, 0) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        resource_schedules = defaultdict(list)
        global_resource_occupations = defaultdict(IntervalTree)
        node_resources = topology[1].nodes

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: taskset_lookup[task.name.split('|')[0]].p))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset.pop(0)
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            start_time = self.get_start_time(next_task, global_resource_occupations, node_resources, max([next_task.a, earliest, last_start]))
            if start_time + next_task.c > next_task.d:
                return None, False

            # Introduce a new instance into the taskset if necessary
            last_task_start[original_taskname] = start_time
            instance = int(instance)

            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: taskset_lookup[task.name.split('|')[0]].p))

            # Add schedule information to resource schedules
            resource_intervals = defaultdict(list)
            task_start = start_time
            task_end = task_start + next_task.c
            interval = (task_start, task_end)
            for resource in next_task.resources:
                resource_intervals[resource].append(interval)
                resource_schedules[resource].append((task_start, task_end, next_task))

            # Add the schedule information to the overall schedule
            schedule.append((start_time, start_time + next_task.c, next_task))

            # Update windowed resource schedules
            if taskset:
                min_chop = max(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                earliest = min_chop
                for resource in next_task.resources:
                    resource_interval_tree = IntervalTree(Interval(begin, end) for begin, end in resource_intervals[resource])
                    global_resource_occupations[resource] |= resource_interval_tree
                    global_resource_occupations[resource].chop(0, min_chop)
                    global_resource_occupations[resource].merge_overlaps(strict=False)

        # Check validity
        valid = True
        for rn, rs in resource_schedules.items():
            for start, end, task in rs:
                if task.d < end:
                    valid = False

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        # Find the earliest start
        offset = self.find_earliest_start(task, resource_occupations, node_resources, earliest)
        return offset

    def find_earliest_start(self, task, resource_occupations, node_resources, earliest):
        start = earliest
        distance_to_free = float('inf')
        while distance_to_free != 0:
            offset = start - task.a
            resource_relations = self.map_task_resources(task, resource_occupations, node_resources, offset)
            task.resources = list(set(resource_relations.values()))

            distance_to_free = 0
            sched_interval = Interval(start, start + task.c)
            for resource in task.resources:
                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start

    def map_task_resources(self, task, resource_occupations, node_resources, offset):
        resource_relations = {}
        for resource in task.resources:
            resource_node, resource_id = resource.split('-')
            resource_type = resource_id[0]
            resource_string = self.get_resource_string(resource)
            if not resource_relations.get(resource_string):
                resource_relations[resource_string] = list(node_resources[resource_node][
                                                               'comm_qs' if resource_type == "C" else "storage_qs"])

        virtual_to_map = {}
        resource_interval_list = [(resource, itree) for resource, itree in task.get_resource_intervals().items()]
        resource_intervals = list(sorted(resource_interval_list, key=lambda ri: ri[1].begin()))
        for resource, itree in resource_intervals:
            offset_itree = IntervalTree([Interval(i.begin + offset, i.end + offset) for i in itree])
            available_resources = self.sort_map_by_availability(resource, resource_relations, resource_occupations,
                                                                offset_itree)
            dist, mapped = available_resources[0]
            virtual_to_map[resource] = mapped
            resource_string = self.get_resource_string(resource)
            resource_relations[resource_string].remove(mapped)

        return virtual_to_map

    def sort_map_by_availability(self, resource, resource_relations, resource_occupations, itree):
        resource_string = self.get_resource_string(resource)
        possible_resources = resource_relations[resource_string]
        available_resources = []
        for pr in possible_resources:
            dist = 0
            for interval in itree:
                intervals = sorted(resource_occupations[pr][interval.begin:interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    dist = max(dist, overlapping_interval.end - interval.begin)
            available_resources.append((dist, pr))

        return list(sorted(available_resources))

    def get_resource_string(self, resource):
        resource_node, resource_id = resource.split('-')
        resource_type = resource_id[0]
        return resource_node + resource_type


class MultipleResourceBlockNPRMScheduler(Scheduler):
    def schedule_tasks(self, dagset, topology):
        # Convert DAGs into tasks
        tasks = {}
        resources = set()
        for dag_task in dagset:
            block_task = PeriodicResourceDAGTask(name=dag_task.name, tasks=dag_task.subtasks, p=dag_task.p)
            tasks[block_task.name] = block_task
            resources |= block_task.resources

        # Separate tasks based on resource requirements
        G = nx.Graph()
        for r in resources:
            G.add_node(r)
        for block_task in tasks.values():
            G.add_node(block_task.name)
            for r in block_task.resources:
                G.add_edge(block_task.name, r)

        sub_graphs = nx.connected_components(G)
        tasksets = []
        for nodes in sub_graphs:
            task_names = nodes - resources
            taskset = [tasks[name] for name in task_names]
            tasksets.append(taskset)

        # For each set of tasks use NPEDFScheduler
        scheduler = MultiResourceBlockNPRMScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset, topology)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules
import networkx as nx
from copy import copy
from collections import defaultdict
from intervaltree import Interval, IntervalTree
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler
from jobscheduling.task import get_lcm_for, ResourceTask, PeriodicResourceDAGTask


logger = LSLogger()


class MultiResourceBlockCEDFScheduler(Scheduler):
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
        last_task_start = defaultdict(int)
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)
        node_resources = topology[1].nodes

        resource_critical_queue = defaultdict(list)
        task_to_max_start = {}

        for task in taskset:
            original_taskname, _ = task.name.split('|')
            task_to_max_start[original_taskname] = task.d - task.c
            entry = (task.d - task.c, task)
            for resource in task.resources:
                resource_critical_queue[resource].append(entry)

        resource_schedules = defaultdict(list)
        global_resource_occupations = defaultdict(IntervalTree)

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: task.d))

        schedule = []
        earliest = 0
        while taskset:
            next_task = taskset[0]
            next_task_original_name, instance = next_task.name.split('|')
            next_task_last_start = last_task_start[next_task_original_name]

            # Help out the scheduling by informing the earliest point at which the new task instance can be scheduled
            next_task_earliest_start = max(next_task.a, next_task_last_start)

            # Find the start time of this task
            next_task_start_time = self.get_start_time(next_task, self.global_resource_occupations, node_resources, next_task_earliest_start)

            # Find the next critical task with earliest that shares resources with next_task
            max_start = float('inf')
            next_critical_task = None
            for resource in next_task.resources:
                ms, nct = resource_critical_queue[resource][0]
                if ms < max_start or next_critical_task is None:
                    max_start = ms
                    next_critical_task = nct

            next_critical_task_original_name, instance = next_critical_task.name.split('|')
            next_critical_task_last_start = last_task_start[next_critical_task_original_name]

            next_task_earliest_start = max(next_critical_task.a, next_critical_task_last_start)
            next_critical_task_start_time = self.get_start_time(next_critical_task, self.global_resource_occupations, node_resources, next_task_earliest_start)


            # Check if starting next_task would cause next_critical_task to miss its latest start time
            # Need to check if there is empty space in resource schedules?
            if start_time + next_task.c > max_start and next_task != next_critical_task and next_critical_task_start_time <= max_start:
                if (start_time + next_task.c) > (next_task.d - next_task.c):
                    # Remove next_task from critical queue
                    max_start = task_to_max_start[original_taskname]
                    for resource in next_task.resources:
                        critical_queue[resource].remove((max_start, next_task))

                    new_max_start = next_task.a + next_task.c
                    # Reinsert with updated max start time
                    critical_queue.append((new_max_start, next_task))
                    task_to_max_start[next_task.name] = new_max_start
                    critical_queue = list(sorted(critical_queue))

                next_task.a = next_critical_task.a + next_critical_task.c
                taskset.append(next_task)
                taskset = list(sorted(taskset, key=lambda task: task.a))




















            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            start_time = self.get_start_time(next_task, global_resource_occupations, max([next_task.a, earliest, last_start]))

            # Introduce a new instance into the taskset if necessary
            last_task_start[original_taskname] = start_time
            instance = int(instance)

            if instance < instance_count[original_taskname]:
                periodic_task = taskset_lookup[original_taskname]
                task_instance = self.create_new_task_instance(periodic_task, instance + 1)
                taskset = list(sorted(taskset + [task_instance], key=lambda task: task.a))

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

    def get_start_time(self, task, resource_occupations, earliest):
        # Find the earliest start
        offset = self.find_earliest_start(task, resource_occupations, earliest)
        return offset

    def find_earliest_start(self, task, resource_occupations, earliest):
        start = earliest
        distance_to_free = float('inf')
        while distance_to_free != 0:
            sched_interval = Interval(start, start + task.c)

            distance_to_free = 0
            for resource in task.resources:
                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultipleResourceBlockCEDFScheduler(Scheduler):
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
        scheduler = MultiResourceBlockNPEDFScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules
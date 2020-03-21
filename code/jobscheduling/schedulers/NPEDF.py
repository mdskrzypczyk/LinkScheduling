import networkx as nx
from copy import copy
from collections import defaultdict
from queue import PriorityQueue
from intervaltree import Interval, IntervalTree
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler, verify_schedule
from jobscheduling.task import get_lcm_for, generate_non_periodic_task_set, ResourceDAGTask, PeriodicResourceDAGTask


logger = LSLogger()


class MultiResourceNPEDFScheduler(Scheduler):
    def preprocess_taskset(self, taskset):
        return taskset

    def create_new_task_instance(self, periodic_task, instance):
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p*instance
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = ResourceDAGTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset,
                                       d=dag_copy.a + dag_copy.p*(instance + 1), tasks=dag_copy.subtasks)
        return dag_instance

    def initialize_taskset(self, tasks):
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    def schedule_tasks(self, taskset, topology=None):
        # Save the original task set
        original_taskset = taskset

        # Compute the hyperperiod needed for the schedule
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))

        # Task lookup when introducing new instance, instance count to track how many instances introduced
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        next_task_release = dict([(t.name, t.a) for t in original_taskset])
        instance_count = dict([(t.name, 0) for t in original_taskset])

        # Initialize the active taskset to one instance of all periodic tasks, track the last time each instance started
        taskset = self.initialize_taskset(taskset)
        last_task_start = defaultdict(int)

        # Track the occupation periods of the resources
        global_resource_occupations = defaultdict(IntervalTree)
        node_resources = topology[1].nodes

        # Sort the initial taskset by deadline
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: task.d))
        ready_queue = PriorityQueue()

        # Track the schedule
        schedule = []

        # Continue scheduling until no more task instances left
        while any([release != float('inf') for release in next_task_release.values()]):
            # Introduce a new task if there are currently none
            if ready_queue.empty():
                name, release = sorted(next_task_release.items(), key=lambda ntr: (ntr[1], ntr[0]))[0]
                periodic_task = taskset_lookup[name]
                instance = instance_count[name]
                task_instance = self.create_new_task_instance(periodic_task, instance)
                if hyperperiod // periodic_task.p == instance_count[name]:
                    next_task_release[name] = float('inf')
                else:
                    instance_count[name] += 1
                    next_task_release[name] += periodic_task.p
                ready_queue.put((task_instance.d, task_instance))

            deadline, next_task = ready_queue.get()
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            # Help out the scheduling by informing the earliest point at which the new task instance can be scheduled
            earliest_start = max(next_task.a, last_start)

            # Find the start time of this task and track it
            start_time = self.get_start_time(next_task, global_resource_occupations, node_resources, earliest_start)
            if start_time + next_task.c > next_task.d:
                return None, False
            last_task_start[original_taskname] = start_time

            # Construct the intervals that the resources are in use by this task
            resource_interval_trees = {}
            resource_interval_list = [(resource, itree) for resource, itree in next_task.get_resource_intervals().items()]
            resource_intervals = list(sorted(resource_interval_list, key=lambda ri: ri[1].begin()))
            for resource, itree in resource_intervals:
                offset_itree = IntervalTree([Interval(i.begin + start_time - next_task.a, i.end + start_time - next_task.a) for i in itree])
                resource_interval_trees[resource] = offset_itree

            # Introduce any new instances that are now available
            for name, release in next_task_release.items():
                for resource, itree in resource_interval_trees.items():
                    periodic_task = taskset_lookup[name]
                    if resource in periodic_task.resources and itree.end() >= release:
                        instance = instance_count[name]
                        task_instance = self.create_new_task_instance(periodic_task, instance)
                        if hyperperiod // periodic_task.p == instance_count[name]:
                            next_task_release[name] = float('inf')
                        else:
                            instance_count[name] += 1
                            next_task_release[name] += periodic_task.p
                        ready_queue.put((task_instance.d, task_instance))
                        break

            # Add the schedule information to the overall schedule
            schedule.append((start_time, start_time + next_task.c, next_task))

            # Update windowed resource schedules
            if taskset:
                min_chop = min(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                for resource in next_task.resources:
                    resource_interval_tree = resource_interval_trees[resource]
                    global_resource_occupations[resource] |= resource_interval_tree
                    global_resource_occupations[resource].chop(0, min_chop)
                    global_resource_occupations[resource].merge_overlaps(strict=False)

        # Check validity
        valid = verify_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        offset = max(0, earliest - task.a)

        # Find the earliest start
        resource_relations = self.map_task_resources(task, resource_occupations, node_resources, offset)
        task.resources = list(set(resource_relations.values()))
        for subtask in task.subtasks:
            new_resources = []
            for resource in subtask.resources:
                new_resources.append(resource_relations[resource])
            subtask.resources = new_resources

        offset = self.find_earliest_start(task, resource_occupations, earliest)
        while True:
            # See if we can schedule now, otherwise get the minimum number of slots forward before the constrained resource can be scheduled
            scheduleable, step = self.attempt_schedule(task, offset, resource_occupations)

            if scheduleable:
                return offset + task.a

            else:
                # See if we can remove any of the constrained subtasks if we have to move forward by step
                offset += step
                resource_relations = self.map_task_resources(task, resource_occupations, node_resources, offset)
                task.resources = list(set(resource_relations.values()))
                for subtask in task.subtasks:
                    new_resources = []
                    for resource in subtask.resources:
                        new_resources.append(resource_relations[resource])
                    subtask.resources = new_resources

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
            offset_itree = IntervalTree([Interval(i.begin+offset, i.end+offset) for i in itree])
            available_resources = self.sort_map_by_availability(resource, resource_relations, resource_occupations, offset_itree)
            dist, mapped = available_resources[0]
            virtual_to_map[resource] = mapped
            resource_string = self.get_resource_string(resource)
            resource_relations[resource_string].remove(mapped)

        return virtual_to_map

    def find_earliest_start(self, task, resource_occupations, start):
        # Iterate over the tasks and check if their interval overlaps with the resources
        distances_to_free = []

        resource_interval_list = [(resource, itree) for resource, itree in task.get_resource_intervals().items()]
        resource_intervals = list(sorted(resource_interval_list, key=lambda ri: ri[1].begin()))
        offset = start-task.a
        for resource, itree in resource_intervals:
            for interval in itree:
                intervals = sorted(resource_occupations[resource][interval.begin + offset:interval.end + offset])
                if intervals:
                    overlapping_interval = intervals[0]
                    distances_to_free.append(overlapping_interval.end - (interval.begin + offset))

        return max([0] + distances_to_free)

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

    def attempt_schedule(self, task, offset, resource_occupations):
        # Iterate over the tasks and check if their interval overlaps with the resources
        distances_to_free = []
        resource_interval_list = [(resource, itree) for resource, itree in task.get_resource_intervals().items()]
        resource_intervals = list(sorted(resource_interval_list, key=lambda ri: ri[1].begin()))
        for resource, itree in resource_intervals:
            for interval in itree:
                intervals = sorted(resource_occupations[resource][interval.begin + offset:interval.end + offset])
                if intervals:
                    overlapping_interval = intervals[0]
                    distances_to_free.append(overlapping_interval.end - (interval.begin + offset))

        if not distances_to_free:
            return True, 0

        else:
            return False, min(distances_to_free)

    def subtask_constrained(self, subtask, resource_occupations, offset):
        # Check if the start time of this resource is affected by the available intervals
        for resource in subtask.resources:
            rightmost_interval = resource_occupations[resource].end()
            if subtask.a + offset < rightmost_interval:
                return True

        return False


class PeriodicMultipleResourceNPEDFScheduler(MultiResourceNPEDFScheduler):
    pass


class MultipleResourceNonBlockNPEDFScheduler(Scheduler):
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
        scheduler = PeriodicMultipleResourceNPEDFScheduler()
        schedules = []
        for taskset in tasksets:
            schedule, valid = scheduler.schedule_tasks(taskset, topology)
            schedules.append((taskset, schedule, valid))

        # Set of schedules is the schedule for each group of resources
        return schedules
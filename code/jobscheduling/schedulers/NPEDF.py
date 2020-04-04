import networkx as nx
from copy import copy
from collections import defaultdict
from queue import PriorityQueue
from intervaltree import Interval, IntervalTree
from jobscheduling.modintervaltree import ModIntervalTree
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler, CommonScheduler, verify_schedule
from jobscheduling.task import get_resource_type_intervals, get_lcm_for, generate_non_periodic_task_set, ResourceDAGTask, PeriodicResourceDAGTask, PeriodicBudgetResourceDAGTask


logger = LSLogger()


class MultiResourceNPEDFScheduler(CommonScheduler):
    def update_resource_occupations(self, resource_occupations, resource_intervals, node_resources):
        for resource in resource_intervals.keys():
            resource_interval_tree = resource_intervals[resource]
            node, rt = resource
            max_capacity = len(node_resources[node]["comm_qs" if rt == "C" else "storage_qs"])
            for interval in resource_interval_tree:
                resource_occupations[resource].add(Interval(interval.begin, interval.end, (interval.data[0], max_capacity)))
            resource_occupations[resource].split_overlaps(data_reducer=lambda x, y: (x[0] + y[0], max_capacity))
            resource_occupations[resource].merge_overlaps(data_reducer=lambda x, y: (x[0], max_capacity),
                                                          data_compare=lambda x, y: x[0] == y[0])

    def extract_resource_intervals(self, next_task, start_time):
        resource_interval_trees = {}
        resource_intervals = get_resource_type_intervals(next_task)
        for resource, itree in resource_intervals.items():
            offset_itree = IntervalTree(
                [Interval(i.begin + start_time - next_task.a, i.end + start_time - next_task.a, i.data) for i in itree])
            resource_interval_trees[resource] = offset_itree

        return resource_interval_trees

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
        global_resource_capacities = defaultdict(ModIntervalTree)
        node_resources = topology[1].nodes

        # Sort the initial taskset by deadline
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: task.d))
        ready_queue = PriorityQueue()

        # Track the schedule
        schedule = []

        import time
        start = time.time()
        # Continue scheduling until no more task instances left
        while any([release != float('inf') for release in next_task_release.values()]):
            # Introduce a new task if there are currently none
            if ready_queue.empty():
                self.populate_ready_queue(ready_queue, taskset_lookup, instance_count, next_task_release, hyperperiod)

            deadline, next_task = ready_queue.get()
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            # Help out the scheduling by informing the earliest point at which the new task instance can be scheduled
            earliest_start = max(next_task.a, last_start)

            # Find the start time of this task and track it
            start_time = self.get_start_time(next_task, global_resource_capacities, node_resources, earliest_start)
            if start_time + next_task.c > next_task.d:
                return None, False
            last_task_start[original_taskname] = start_time

            # Construct the intervals that the resources are in use by this task
            resource_interval_trees = self.extract_resource_intervals(next_task, start_time)

            # Introduce any new instances that are now available
            self.check_for_released_tasks(ready_queue, resource_interval_trees, taskset_lookup, instance_count, next_task_release, hyperperiod)

            # Add the schedule information to the overall schedule
            schedule.append((start_time, start_time + next_task.c, next_task))

            # Update windowed resource schedules
            if taskset:
                self.update_resource_occupations(global_resource_capacities, resource_interval_trees, node_resources)
                min_chop = min(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                self.remove_useless_resource_occupations(global_resource_capacities, resource_interval_trees.keys(), min_chop)

        print("Scheduling took {}".format(time.time() - start))
        # Check validity
        valid = verify_schedule(original_taskset, schedule, topology)

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_capacities, node_resources, earliest):
        offset = max(0, earliest - task.a)
        offset = self.find_earliest_start(task, resource_capacities, earliest)
        while True:
            # See if we can schedule now, otherwise get the minimum number of slots forward before the constrained resource can be scheduled
            scheduleable, step = self.attempt_schedule(task, offset, resource_capacities)
            if scheduleable:
                return offset + task.a

            else:
                # See if we can remove any of the constrained subtasks if we have to move forward by step
                offset += step

    def find_earliest_start(self, task, resource_capacities, start):
        # Iterate over the tasks and check if their interval overlaps with the resources
        distances_to_free = []

        resource_intervals = get_resource_type_intervals(task)
        offset = start-task.a
        for resource, itree in resource_intervals.items():
            for interval in itree:
                intervals = sorted(resource_capacities[resource][interval.begin + offset:interval.end + offset])
                for overlapping_interval in intervals:
                    if interval.data[0] + overlapping_interval.data[0] > overlapping_interval.data[1]:
                        distances_to_free.append(overlapping_interval.end - (interval.begin + offset))

        return max([0] + distances_to_free)

    def attempt_schedule(self, task, offset, resource_capacities):
        # Iterate over the tasks and check if their interval overlaps with the resources
        distances_to_free = []
        resource_intervals = get_resource_type_intervals(task)
        for resource, itree in resource_intervals.items():
            for interval in itree:
                intervals = sorted(resource_capacities[resource][interval.begin + offset:interval.end + offset])
                for overlapping_interval in intervals:
                    if interval.data[0] + overlapping_interval.data[0] > overlapping_interval.data[1]:
                        distances_to_free.append(overlapping_interval.end - (interval.begin + offset))

        if not distances_to_free:
            return True, 0

        else:
            return False, max(distances_to_free)

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
            block_task = PeriodicBudgetResourceDAGTask(name=dag_task.name, tasks=dag_task.subtasks, p=dag_task.p,
                                                       k=int(dag_task.k))
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
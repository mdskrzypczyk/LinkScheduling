from collections import defaultdict
from intervaltree import Interval, IntervalTree
from queue import PriorityQueue
from jobscheduling.log import LSLogger
from jobscheduling.schedulers.scheduler import Scheduler, CommonScheduler, BaseMultipleResourceScheduler,\
    verify_schedule
from jobscheduling.task import get_lcm_for


logger = LSLogger()


class UniResourceBlockNPEDFScheduler(Scheduler):
    """
    Uniprocessor NP-EDF scheduler
    """
    def schedule_tasks(self, taskset, topology=None):
        """
        Main scheduling function for uniprocessor NP-EDF
        :param taskset: type list
            List of PeriodicTasks to schedule
        :param topology: tuple
            Tuple of networkx.Graphs that represent the communication resources and connectivity graph of the network
        :return: list
            Contains a tuple of (taskset, schedule, valid) where valid indicates if the schedule is valid
        """
        original_taskset = taskset
        taskset = self.preprocess_taskset(taskset)
        queue = PriorityQueue()
        schedule = []

        # First sort the taskset by activation time
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        instance_count = dict([(t.name, hyperperiod // t.p - 1) for t in original_taskset])
        taskset = self.initialize_taskset(taskset)
        taskset = sorted(taskset, key=lambda task: (task.a, task.d))

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
                    taskset = list(sorted(taskset + [task_instance], key=lambda task: (task.a, task.d)))

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


class MultiResourceBlockNPEDFScheduler(CommonScheduler):
    def extract_resource_intervals(self, next_task, start_time):
        """
        Obtains the intervals of task execution
        :param next_task: type Task
            The task to obtain the resource intervals for
        :return: type int
            The start time of the occupation
        """
        resource_interval_trees = {}
        task_start = start_time
        task_end = task_start + next_task.c
        for resource in next_task.resources:
            resource_interval_trees[resource] = IntervalTree([Interval(task_start, task_end, next_task)])

        return resource_interval_trees

    def update_resource_occupations(self, resource_occupations, resource_intervals):
        """
        Updates the recorded occupation of resources in the system
        :param resource_occupations: type dict(IntervalTree)
            A dictionary from resource identifiers to interval trees representing the periods of occupation of resources
        :param resource_intervals: type dict(IntervalTree)
            A dictionary of the newly occupied time intervals to add to the recorded occupation
        :return: None
        """
        for resource in resource_intervals.keys():
            resource_interval_tree = resource_intervals[resource]
            resource_occupations[resource] |= resource_interval_tree
            resource_occupations[resource].merge_overlaps(strict=False)

    def schedule_tasks(self, taskset, topology):
        """
        Main scheduling function for RCPSP NP-FPR
        :param taskset: type list
            List of PeriodicTasks to schedule
        :param topology: tuple
            Tuple of networkx.Graphs that represent the communication resources and connectivity graph of the network
        :return: list
            Contains a tuple of (taskset, schedule, valid) where valid indicates if the schedule is valid
        """
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

        while any([release != float('inf') for release in next_task_release.values()]) or not ready_queue.empty():
            # Introduce a new task if there are currently none
            if ready_queue.empty():
                self.populate_ready_queue(ready_queue, taskset_lookup, instance_count, next_task_release, hyperperiod)

            deadline, next_task = ready_queue.get()
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            # Help out the scheduling by informing the earliest point at which the new task instance can be scheduled
            earliest_start = max(next_task.a, last_start)

            start_time = self.get_start_time(next_task, global_resource_occupations, node_resources, earliest_start)
            if start_time + next_task.c > next_task.d:
                return None, False
            last_task_start[original_taskname] = start_time

            # Construct the intervals that the resources are in use by this task
            resource_interval_trees = self.extract_resource_intervals(next_task, start_time)

            # Introduce any new instances that are now available
            self.check_for_released_tasks(ready_queue, resource_interval_trees, taskset_lookup, instance_count,
                                          next_task_release, hyperperiod)

            # Add the schedule information to the overall schedule
            schedule.append((start_time, start_time + next_task.c, next_task))

            # Update windowed resource schedules
            if taskset:
                self.update_resource_occupations(global_resource_occupations, resource_interval_trees)
                min_chop = min(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                self.remove_useless_resource_occupations(global_resource_occupations, resource_interval_trees.keys(),
                                                         min_chop)

        # Check validity
        valid = verify_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid

    def get_start_time(self, task, resource_occupations, node_resources, earliest):
        """
        Obtains the start time for a task
        :param task: type DAGTask
            The task to obtain the start time for
        :param resource_occupations: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing resource occupations
        :param node_resources: type dict
            A dictionary of node to resource identifiers held by a node
        :param earliest: type int
            The earliest point in time the task is permitted to start
        :return: type int
            The earliest time the incoming task may start
        """
        # Find the earliest start
        offset = self.find_earliest_start(task, resource_occupations, node_resources, earliest)
        return offset

    def find_earliest_start(self, task, resource_occupations, node_resources, earliest):
        """
        Finds the start time for a task
        :param task: type DAGTask
            The task to obtain the start time for
        :param resource_occupations: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing resource occupations
        :param node_resources: type dict
            A dictionary of node to resource identifiers held by a node
        :param earliest: type int
            The earliest point in time the task is permitted to start
        :return: type int
            The earliest time the incoming task may start
        """
        start = earliest
        distance_to_free = float('inf')
        while distance_to_free != 0:
            distance_to_free = 0
            sched_interval = Interval(start, start + task.c)
            for resource in task.resources:
                intervals = sorted(resource_occupations[resource][sched_interval.begin:sched_interval.end])
                if intervals:
                    overlapping_interval = intervals[0]
                    distance_to_free = max(distance_to_free, overlapping_interval.end - start)

            start += distance_to_free

        return start


class MultipleResourceBlockNPEDFScheduler(BaseMultipleResourceScheduler):
    internal_scheduler_class = MultiResourceBlockNPEDFScheduler

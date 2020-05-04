from abc import abstractmethod
from collections import defaultdict
from copy import copy
from queue import PriorityQueue
from jobscheduling.task import find_dag_task_preemption_points, get_lcm_for, BudgetResourceDAGTask, BudgetResourceTask
from jobscheduling.log import LSLogger
from intervaltree import IntervalTree, Interval


logger = LSLogger()


def verify_schedule(original_taskset, schedule):
    """
    Verifies that a schedule adheres to task constraints and respects resource utilization
    :param original_taskset: type list
        List of the periodic tasks that were scheduled
    :param schedule: list of tuples
        Tuples describe start, end, task information of schedule
    :return: bool
        True/False
    """
    # Construct the occupation intervals of all the resources
    global_resource_intervals = defaultdict(IntervalTree)

    # Iterate over the chedule
    for start, end, t in schedule:
        # Check that the task's execution period adhere's to it's release and deadline times
        if start < t.a or end > t.d:
            logger.warning("Found task {} ({}, {}) that does not adhere"
                           "to release/deadline constraints ({}, {})".format(t.name, start, end, t.a, t.d))
            return False

        # Check that the start and end periods align with the tasks runtime
        if end - start != t.c:
            logger.warning("Found task {} that does not have start/end corresponding to duration".format(t.name))
            return False

        # Add the occupation period of this task to all resources
        task_resource_intervals = t.get_resource_intervals()
        offset = start - t.a
        for resource, itree in task_resource_intervals.items():
            offset_itree = IntervalTree([Interval(i.begin + offset, i.end + offset, t) for i in itree])
            for interval in offset_itree:
                if global_resource_intervals[resource].overlap(interval.begin, interval.end):
                    import pdb
                    pdb.set_trace()
                    return False
                global_resource_intervals[resource].add(interval)

    return True


def verify_budget_schedule(original_taskset, schedule):
    """
    Verifies that a schedule adheres to task constraints including preemption budget and respects resource utilization
    :param original_taskset: type list
        List of the periodic tasks that were scheduled
    :param schedule: list of tuples
        Tuples describe start, end, task information of schedule
    :return: bool
        True/False
    """
    # Construct the occupation intervals of all the resources
    global_resource_intervals = defaultdict(IntervalTree)
    taskset_lookup = dict([(t.name, t) for t in original_taskset])
    task_starts = {}
    task_ends = {}
    task_exec_times = defaultdict(int)

    # Iterate over the schedule
    for start, end, t in schedule:
        original_taskname, instance = t.name.split('|')[0:2]
        instance_name = "|".join([original_taskname, instance])

        if not task_starts.get(instance_name):
            task_starts[instance_name] = start
        task_ends[instance_name] = end

        task_exec_times[instance_name] += end - start
        if task_exec_times[instance_name] > taskset_lookup[original_taskname].c:
            import pdb
            pdb.set_trace()

        elif task_exec_times[instance_name] == taskset_lookup[original_taskname].c:
            task_exec_times.pop(instance_name)
            if task_ends[instance_name] - task_starts[instance_name] > \
                    taskset_lookup[original_taskname].k + taskset_lookup[original_taskname].c:
                logger.warning("Task {} does not adhere to budget constraints".format(t.name))
                return False

        # Check that the task's execution period adhere's to it's release and deadline times
        if start < t.a or end > t.d:
            logger.warning("Found task {} ({}, {}) that does not adhere"
                           "to release/deadline constraints ({}, {})".format(t.name, start, end, t.a, t.d))
            return False

        # Add the occupation period of this task to all resources
        task_resource_intervals = t.get_resource_intervals()
        offset = start - t.a
        for resource, itree in task_resource_intervals.items():
            offset_itree = IntervalTree([Interval(start, end, t) for i in itree
                                         if start <= i.begin + offset < end and start < i.end + offset <= end])
            for interval in offset_itree:
                if global_resource_intervals[resource].overlap(interval.begin, interval.end):
                    import pdb
                    pdb.set_trace()
                    return False
                global_resource_intervals[resource].add(interval)

    # Check that the start and end periods align with the tasks runtime
    if task_exec_times != {}:
        import pdb
        pdb.set_trace()
        return False

    return True


def verify_segmented_budget_schedule(original_taskset, schedule):
    """
   Verifies that a schedule adheres to task constraints including preemption budget and respects resource utilization.
   Specifically for the case of segmented protocols.
   :param original_taskset: type list
       List of the periodic tasks that were scheduled
   :param schedule: list of tuples
       Tuples describe start, end, task information of schedule
   :return: bool
       True/False
   """
    # Construct the occupation intervals of all the resources
    global_resource_intervals = defaultdict(IntervalTree)
    taskset_lookup = dict([(t.name, t) for t in original_taskset])
    task_starts = {}
    task_ends = {}
    task_exec_times = defaultdict(int)

    # Iterate over the schedule
    for start, end, t in schedule:
        original_taskname, instance, segment = t.name.split('|')
        instance_name = "|".join([original_taskname, instance])

        if not task_starts.get(instance_name):
            task_starts[instance_name] = start
        task_ends[instance_name] = end

        task_exec_times[instance_name] += end - start
        if task_exec_times[instance_name] > taskset_lookup[original_taskname].c:
            import pdb
            pdb.set_trace()

        elif task_exec_times[instance_name] == taskset_lookup[original_taskname].c:
            task_exec_times.pop(instance_name)
            if task_ends[instance_name] - task_starts[instance_name] > \
                    taskset_lookup[original_taskname].k + taskset_lookup[original_taskname].c:
                logger.warning("Task {} does not adhere to budget constraints".format(instance_name))
                return False

        # Check that the task's execution period adhere's to it's release and deadline times
        if start < t.a or end > t.d:
            logger.warning("Found task {} ({}, {}) that does not adhere"
                           "to release/deadline constraints ({}, {})".format(instance_name, start, end, t.a, t.d))
            return False

        # Add the occupation period of this task to all resources
        task_resource_intervals = t.get_resource_intervals()
        offset = start - t.a
        for resource, itree in task_resource_intervals.items():
            offset_itree = IntervalTree([Interval(i.begin + offset, i.end + offset, t) for i in itree
                                         if start <= i.begin + offset < end and start < i.end + offset <= end])
            for interval in offset_itree:
                if global_resource_intervals[resource].overlap(interval.begin, interval.end):
                    import pdb
                    pdb.set_trace()
                    return False
                global_resource_intervals[resource].add(interval)

    # Check that the start and end periods align with the tasks runtime
    if task_exec_times != {}:
        import pdb
        pdb.set_trace()
        return False

    return True


class Scheduler:
    def __init__(self):
        """
        Base class for periodic task schedulers
        """
        self.curr_time = 0
        self.schedule = None
        self.taskset = None

    def preprocess_taskset(self, taskset):
        """
        Overridable method for preprocessing the taskset
        :param taskset: type list
            List of Tasks to preprocess
        :return: type list
            The processed taskset
        """
        return taskset

    def add_to_schedule(self, task, duration):
        """
        Adds an entry to the internal schedule
        :param task: type Task
            The task to schedule
        :param duration: type int
            The duration the task executes
        :return:
        """
        self.schedule.append((self.curr_time, self.curr_time + duration, task))
        self.curr_time += duration

    def create_new_task_instance(self, periodic_task, instance):
        """
        Creates a new task instance for a periodic task
        :param periodic_task: type PeriodicTask
            The periodic task to produce an instance for
        :param instance: type int
            The instance number of the new instance
        :return: type Task
            The new task instance
        """
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p * instance
        dag_copy.a = release_offset
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset
        dag_instance = BudgetResourceTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset, c=dag_copy.c,
                                          d=release_offset + dag_copy.p, resources=dag_copy.resources,
                                          k=dag_copy.k, preemption_points=find_dag_task_preemption_points(dag_copy))

        return dag_instance

    def initialize_taskset(self, tasks):
        """
        Produces the initial set of tasks to be scheduled
        :param tasks: type list
            List of the PeriodicTasks to produce intiial instances for
        :return: type list
            List of the intiial task instances
        """
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    @abstractmethod
    def schedule_tasks(self, taskset):
        """
        Main method for scheduling a taskset
        :param taskset: type list
            List of Tasks to schedule
        """
        pass


class CommonScheduler:
    """
    Base class for RCPSP Schedulers
    """
    def preprocess_taskset(self, taskset):
        """
        Overridable method for preprocessing the taskset
        :param taskset: type list
            List of Tasks to preprocess
        :return: type list
            The processed taskset
        """
        return taskset

    def initialize_taskset(self, tasks):
        """
        Produces the initial set of tasks to be scheduled
        :param tasks: type list
            List of the PeriodicTasks to produce intiial instances for
        :return: type list
            List of the intiial task instances
        """
        initial_tasks = []
        for task in tasks:
            task_instance = self.create_new_task_instance(task, 0)
            initial_tasks.append(task_instance)

        return initial_tasks

    def check_for_released_tasks(self, ready_queue, resource_intervals, taskset_lookup, instance_count,
                                 next_task_release, hyperperiod):
        """
        Checks if there are any new tasks to add into the system
        :param ready_queue: type PriorityQueue
            A priority queue of the task instances released into the system
        :param resource_intervals: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees where the resources are occupied
        :param taskset_lookup: type dict
            A dictionary of original periodic task names to original periodic tasks
        :param instance_count: type dict
            A dictionary that tracks the number of task instances have been produced for every periodic tasks
        :param next_task_release: type dict
            A dictionary of original periodic task names to the time the next task instance is released
        :param hyperperiod: type int
            The hyperperiod of the periodic task periods
        :return: None
        """
        for name, release in next_task_release.items():
            for resource, itree in resource_intervals.items():
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

    def populate_ready_queue(self, ready_queue, taskset_lookup, instance_count, next_task_release, hyperperiod):
        """
        Adds newly released task instances to the ready queue
        :param ready_queue: type PriorityQueue
            A priority queue of the task instances released into the system
        :param resource_intervals: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees where the resources are occupied
        :param taskset_lookup: type dict
            A dictionary of original periodic task names to original periodic tasks
        :param instance_count: type dict
            A dictionary that tracks the number of task instances have been produced for every periodic tasks
        :param next_task_release: type dict
            A dictionary of original periodic task names to the time the next task instance is released
        :param hyperperiod: type int
            The hyperperiod of the periodic task periods
        :return: None
        """
        incoming_tasks = list(sorted(next_task_release.items(), key=lambda ntr: (ntr[1], ntr[0])))
        _, next_release = incoming_tasks[0]
        for name, release in incoming_tasks:
            if release != next_release:
                break
            periodic_task = taskset_lookup[name]
            instance = instance_count[name]
            task_instance = self.create_new_task_instance(periodic_task, instance)
            if hyperperiod // periodic_task.p == instance_count[name]:
                next_task_release[name] = float('inf')
            else:
                instance_count[name] += 1
                next_task_release[name] += periodic_task.p
            ready_queue.put((task_instance.d, task_instance))

    def create_new_task_instance(self, periodic_task, instance):
        """
        Creates a new task instance for a periodic task
        :param periodic_task: type PeriodicTask
            The periodic task to produce an instance for
        :param instance: type int
            The instance number of the new instance
        :return: type Task
            The new task instance
        """
        dag_copy = copy(periodic_task)
        release_offset = dag_copy.a + dag_copy.p * instance
        for subtask in dag_copy.subtasks:
            subtask.a += release_offset

        dag_instance = BudgetResourceDAGTask(name="{}|{}".format(dag_copy.name, instance), a=release_offset,
                                             d=dag_copy.a + dag_copy.p * (instance + 1), tasks=dag_copy.subtasks,
                                             k=periodic_task.k)

        return dag_instance

    def remove_useless_resource_occupations(self, resource_occupations, resources, chop):
        """
        Removes data of occupied resource intervals to reduce height of interal trees
        :param resource_occupations: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing the occupation of the resource.
        :param resources: type list
            List of the resources to update the interval trees for
        :param chop: type int
            The lower bound of interval data to keep
        :return: None
        """
        for resource in resources:
            resource_occupations[resource].chop(0, chop)

    def get_segment_tasks(self, task):
        """
        Segments a task representing a repeater protocol into non-overlapping segments
        :param task: type DAGTask
            The task instance to segment
        :return: type list
            List of Tasks that represent the segments of the DAGTask
        """
        segment_tasks = []
        segment_info = find_dag_task_preemption_points(task)
        comp_time = 0
        for i, segment in enumerate(segment_info):
            segment_task = self.construct_segment_task(task, segment, comp_time, i)
            segment_tasks.append(segment_task)
            comp_time += segment_task.c

        return segment_tasks

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
        segment_earliest = earliest
        # Find the earliest start

        while True:
            segment_tasks = self.get_segment_tasks(task)

            segment_intervals = []
            attempt_earliest = segment_earliest
            comp_time = 0
            for segment_task in segment_tasks:
                # Find the earliest start
                segment_start = self.find_earliest_start(segment_task, resource_occupations, attempt_earliest)
                if not segment_intervals and segment_start != segment_earliest:
                    segment_earliest = segment_start
                    break

                attempt_earliest = segment_start + segment_task.c
                comp_time += segment_task.c
                segment_intervals.append((segment_start, segment_start + segment_task.c))
                if segment_start + segment_task.c - segment_intervals[0][0] > comp_time + task.k:
                    segment_earliest = segment_start - comp_time + segment_task.c - task.k
                    break

                elif len(segment_intervals) == len(segment_tasks):
                    start_times = list(zip(segment_intervals, [segment_task for segment_task in segment_tasks]))
                    return start_times

    def schedule_tasks(self, taskset, topology):
        """
        Main method for scheduling a set of periodic tasks to a quantum network specified by the topology
        :param taskset: type list
            List of PeriodicDAGTasks to schedule into the network
        :param topology: type tuple
            Tuple of networkx.Graphs that represent the communication resources and connectivity of the quantum network
        :return:
        """
        original_taskset = taskset
        hyperperiod = get_lcm_for([t.p for t in original_taskset])
        logger.debug("Computed hyperperiod {}".format(hyperperiod))
        taskset_lookup = dict([(t.name, t) for t in original_taskset])
        last_task_start = dict([(t.name, 0) for t in original_taskset])
        next_task_release = dict([(t.name, t.a) for t in original_taskset])
        instance_count = dict([(t.name, 0) for t in original_taskset])

        taskset = self.initialize_taskset(taskset)

        global_resource_occupations = defaultdict(IntervalTree)
        node_resources = topology[1].nodes

        # First sort the taskset by activation time
        logger.debug("Sorting tasks by earliest deadlines")
        taskset = list(sorted(taskset, key=lambda task: (task.d, task.a, task.name)))
        ready_queue = PriorityQueue()

        schedule = []
        earliest = 0
        while any([release != float('inf') for release in next_task_release.values()]):
            # Introduce a new task if there are currently none
            if ready_queue.empty():
                self.populate_ready_queue(ready_queue, taskset_lookup, instance_count, next_task_release, hyperperiod)

            deadline, next_task = ready_queue.get()
            original_taskname, instance = next_task.name.split('|')
            last_start = last_task_start[original_taskname]

            preemption_point_intervals = self.get_start_time(next_task, global_resource_occupations, node_resources,
                                                             max([next_task.a, earliest, last_start]))

            # Check if time violated deadline
            last_pp_interval, last_pp_tasks = preemption_point_intervals[-1]
            last_pp_start, last_pp_end = last_pp_interval
            if last_pp_end > next_task.d:
                return None, False

            # Record start time
            first_interval, _ = preemption_point_intervals[0]
            start_time = first_interval[0]
            last_task_start[original_taskname] = start_time

            # Add schedule information to resource schedules
            resource_intervals = self.extract_resource_intervals_from_preemption_point_intervals(preemption_point_intervals)
            self.schedule_preemption_point_intervals(schedule, preemption_point_intervals)

            # Introduce any new instances that are now available
            self.check_for_released_tasks(ready_queue, resource_intervals, taskset_lookup, instance_count,
                                          next_task_release, hyperperiod)

            # Update windowed resource schedules
            if taskset:
                self.update_resource_occupations(global_resource_occupations, resource_intervals)
                min_chop = max(min(list(last_task_start.values())), min(list([t.a for t in taskset])))
                self.remove_useless_resource_occupations(global_resource_occupations, resource_intervals.keys(),
                                                         min_chop)

        # Check validity
        valid = self.verify_schedule(original_taskset, schedule)

        taskset = original_taskset
        return schedule, valid

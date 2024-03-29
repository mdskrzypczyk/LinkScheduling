from functools import reduce
from math import gcd, ceil
from jobscheduling.log import LSLogger
from collections import defaultdict
from copy import copy
from intervaltree import IntervalTree, Interval

logger = LSLogger()


def lcm(a, b):
    """
    Obtains the least common multiple of to integers
    :param a: type int
    :param b: type int
    :return: type int
    """
    return a * b // gcd(a, b)


def get_lcm_for(values):
    """
    Obtains the least common multiple of a list of integers
    :param values: type list of int
    :return: type int
    """
    return reduce(lambda x, y: lcm(x, y), values)


def gcd_rationals(x, y):
    """
    Computes the greatest common divisr of a pair of rational numbers
    :param x: type float
    :param y: type float
    :return: type float
    """
    a = x
    b = 1
    while int(a) != b * x:
        a *= 10
        b *= 10

    c = y
    d = 1
    while int(c) != d * y:
        c *= 10
        d *= 10

    return gcd(int(a * d), int(c * b)) / (b * d)


def get_gcd_for(values):
    """
    Obtains the greatest common divisor for a list of rational numbers
    :param values: type list of float
    :return: type float
    """
    return reduce(lambda x, y: gcd_rationals(x, y), values)


def get_dag_exec_time(dag):
    """
    Obtains the execution time of DAGTask
    :param dag: type DAGTask
        The DAGTask to obtain the execution time of
    :return: type int
    """
    if not dag.subtasks:
        logger.error("Cannot calculate execution time for DAGTask without subtasks")

    # Get the max finish time of all subtasks
    return max([subtask.a + subtask.c for subtask in dag.subtasks]) - min([source.a for source in dag.sources])


def find_dag_task_preemption_points(budget_dag_task, resources=None):
    """
    Finds the preemption points of a DAGTask
    :param budget_dag_task: type DAGTask
        The DAGTask to find preemption points of
    :param resources: type list
        List of resources to find preemption points over
    :return:
    """
    if resources is None:
        resources = budget_dag_task.resources

    resource_intervals = budget_dag_task.get_resource_intervals(separate_occupation=True)
    global_itree = IntervalTree()

    # Make one interval tree where any resource is in use
    for resource in resources:
        global_itree |= resource_intervals[resource]

    # Merge intervals
    global_itree.merge_overlaps()

    # Preemption points are then the spaces between intervals in global_itree
    preemption_points = list(sorted([(interval.begin, interval.end) for interval in global_itree
                                     if interval.end <= budget_dag_task.a + budget_dag_task.c]))

    # Now check which resources are locked at each preemption point as well as which
    # resources are needed for the subsequent interval to execute
    points_to_locked_resources = defaultdict(list)
    points_to_needed_resources = defaultdict(list)
    points_to_subtasks = defaultdict(set)

    # Iterate over the set of preemption points
    for point_start, point_end in preemption_points:
        # Iterate over resources
        for resource, itree in resource_intervals.items():
            # Grab all the intervals of the task up until this end point
            locking_intervals = sorted(itree[0:point_end])

            # Check that point_end is not just the end of the task
            if locking_intervals and point_end < budget_dag_task.a + budget_dag_task.c:
                # Get the last task for this resource and check if it is locked
                last_task = locking_intervals[-1].data
                if last_task.locked_resources and resource in last_task.locked_resources:
                    points_to_locked_resources[(point_start, point_end)].append(resource)

            # Check if this resource is used in the rest of the task
            active_intervals = sorted(itree[point_start:budget_dag_task.a + budget_dag_task.c])
            if active_intervals:
                # Record that this resource is needed for the rest of the task
                points_to_needed_resources[(point_start, point_end)].append(resource)

                # Record the subtasks that are active during this interval
                active_intervals = sorted(itree[point_start:point_end])
                for interval in active_intervals:
                    if type(interval.data) != ResourceTask:
                        points_to_subtasks[(point_start, point_end)] |= {interval.data}

    # Store preemption point info with (start, end) points, locked resources, resources needed to continue,
    # and subtasks that belong to this interval
    preemption_points = [(point, points_to_locked_resources[point], points_to_needed_resources[point],
                          points_to_subtasks[point]) for point in preemption_points]

    # Check that all the subtasks are accounted for
    all_pp_subtasks = list()
    for taskset in points_to_subtasks.values():
        all_pp_subtasks += list(filter(lambda task: type(task) != ResourceTask, taskset))

    if len(all_pp_subtasks) != len(budget_dag_task.subtasks) or \
            any([t not in all_pp_subtasks for t in budget_dag_task.subtasks]):
        logger.error("Incorrect subtask set")

    return preemption_points


class Task:
    def __init__(self, name, c, a=0, d=None, description=None, **kwargs):
        """
        Basic task class, has a name, execution time (c), absolute release time (a), and absolute deadline (d). The
        description can be used to encode other information
        :param name: type str
            The name of the task
        :param c: type int
            The number of time units needed to execute the task
        :param a: type int
            The time unit at which the task becomes available
        :param d: type int
            The time unit at which the task is due
        :param description: type obj
            Any information to attach to the task
        :param kwargs:
        """
        self.name = name
        self.a = a
        self.c = c
        self.d = d
        self.description = description

    def __lt__(self, other):
        """
        Comparison of tasks, just by name
        :param other: type Task
            The task to compare to
        :return: type bool
            Whether this task is less than the other task
        """
        return self.name < other.name

    def __copy__(self):
        """
        Used to create a copy of the task
        :return: type Task
            A instance of a Task object with the same name, release, computation time, and deadline
        """
        return Task(name=self.name, a=self.a, c=self.c, d=self.d)


class BudgetTask(Task):
    def __init__(self, name, c, a=0, d=None, k=0, preemption_points=None):
        """
        Budget task class, has a name, execution time (c), release time (a), deadline (d), and preemption budget(k).
        Additionally specifies a set of preemption points where preemption is allowed.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete task
        :param a: type int
            The time unit at which the task becomes available
        :param d: type int
            The time unit at which the task is due
        :param k: type int
            The number of time units the task may spend preempted
        :param preemption_points: type list
            List of preemption points
        """
        super(BudgetTask, self).__init__(name=name, c=c, a=a, d=d)
        self.k = k
        self.preemption_points = preemption_points

    def __copy__(self):
        return BudgetTask(name=self.name, c=self.c, a=self.a, d=self.d, k=self.k,
                          preemption_points=self.preemption_points)


class ResourceTask(Task):
    def __init__(self, name, c, a=0, d=None, resources=None, locked_resources=None):
        """
        Resource task class, similar to a Task but additionally specifies a set of resources needed to execute as
        well as a set of resources that are locked after execution.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete task
        :param a: type int
            The time unit at which the task becomes available
        :param d: type int
            The time unit at which the task is due
        :param resources: type list
            List of resource identifiers (any object)
        :param locked_resources: type list
            List of resource identifiers (any object)
        """
        super(ResourceTask, self).__init__(name=name, c=c, a=a, d=d)
        if resources is None:
            resources = []
        self.resources = resources
        if locked_resources is None:
            locked_resources = []
        self.locked_resources = locked_resources

    def __copy__(self):
        return ResourceTask(name=self.name, c=self.c, a=self.a, d=self.d, resources=self.resources,
                            locked_resources=self.locked_resources)

    def get_resource_schedules(self):
        """
        :return: type dict
            A dictionary of resource identifier to set of slots where the resource is in use.
        """
        resource_schedules = defaultdict(list)
        slots = [(self.a + i, self) for i in range(ceil(self.c))]
        for resource in self.resources:
            resource_schedules[resource] += slots
        return dict(resource_schedules)

    def get_resource_intervals(self):
        """
        :return: type dict
            A dictionary of resource identifier to an interval tree containing intervals where the resource is in use.
        """
        resource_intervals = defaultdict(IntervalTree)
        if self.c > 0:
            interval = Interval(self.a, self.a + self.c)
            for resource in self.resources:
                resource_intervals[resource].add(interval)
        return resource_intervals


class BudgetResourceTask(ResourceTask):
    def __init__(self, name, c, a=0, d=None, k=0, preemption_points=None, resources=None, locked_resources=None):
        """
        Combination of a BudgetTask and a ResourceTask. Can specify all the parameters available to these objects.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete task
        :param a: type int
            The time unit at which the task becomes available
        :param d: type int
            The time unit at which the task is due
        :param k: type int
            The number of time units the task may spend preempted
        :param preemption_points: type list
            A list of preemption point information
        :param resources: type list
            List of resource identifiers (any object)
        :param locked_resources: type list
            List of resource identifiers (any object)
        """
        super(BudgetResourceTask, self).__init__(name=name, c=c, a=a, d=d, resources=resources,
                                                 locked_resources=locked_resources)
        self.k = k
        self.preemption_points = preemption_points

    def __copy__(self):
        return BudgetResourceTask(name=self.name, c=self.c, a=self.a, d=self.d, k=self.k,
                                  preemption_points=self.preemption_points, resources=self.resources,
                                  locked_resources=self.locked_resources)


class PeriodicTask(Task):
    def __init__(self, name, c, a=0, p=None):
        """
        Basic Periodic Task class. Specifies a name, execution time, release offset, and a period at which it is
        released into the system.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete an instance of the task
        :param a: type int
            The time unit at which the first task instance becomes available
        :param p: type int
            The number of time units between periodic releases of the task
        """
        super(PeriodicTask, self).__init__(name=name, a=a, c=c)
        self.p = p

    def __copy__(self):
        return PeriodicTask(name=self.name, a=self.a, c=self.c, p=self.p)


class PeriodicBudgetTask(BudgetTask):
    def __init__(self, name, c, a=0, p=None, k=0, preemption_points=None):
        """
        Combination of a PeriodicTask and a BudgetTask.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete an instance of the task
        :param a: type int
            The time unit at which the first task instance becomes available
        :param p: type int
            The number of time units between periodic releases of the task
        :param k: type int
            The number of time units an instance of the periodic task may spend preempted
        :param preemption_points: type list
            List of preemption points
        """
        super(PeriodicBudgetTask, self).__init__(name=name, a=a, c=c, k=k, preemption_points=preemption_points)
        self.p = p

    def __copy__(self):
        return PeriodicBudgetTask(name=self.name, c=self.c, a=self.a, p=self.p, k=self.k,
                                  preemption_points=self.preemption_points)


class PeriodicResourceTask(ResourceTask):
    def __init__(self, name, c, a=0, p=None, resources=None, locked_resources=None):
        """
        Combination of a PeriodicTask and a ResourceTask.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete an instance of the task
        :param a: type int
            The time unit at which the first task instance becomes available
        :param p: type int
            The number of time units between periodic releases of the task
        :param resources: type list
            List of resource identifiers (any object)
        :param locked_resources: type list
            List of resource identifiers (any object)
        """
        super(PeriodicResourceTask, self).__init__(name=name, a=a, c=c, resources=resources,
                                                   locked_resources=locked_resources)
        self.p = p

    def __copy__(self):
        return PeriodicResourceTask(name=self.name, c=self.c, resources=self.resources,
                                    locked_resources=self.locked_resources)


class PeriodicBudgetResourceTask(PeriodicResourceTask):
    def __init__(self, name, c, a=0, p=None, k=0, preemption_points=None, resources=None, locked_resources=None):
        """
        Combination of a PeriodicTask, BudgetTask, and ResourceTask.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete an instance of the task
        :param a: type int
            The time unit at which the first task instance becomes available
        :param p: type int
            The number of time units between periodic releases of the task
        :param k: type int
            The number of time units an instance of the periodic task may spend preempted
        :param preemption_points: type list
            List of preemption points
        :param resources: type list
            List of resource identifiers (any object)
        :param locked_resources: type list
            List of resource identifiers (any object)
        """
        super(PeriodicBudgetResourceTask, self).__init__(name=name, c=c, a=a, p=p, resources=resources,
                                                         locked_resources=locked_resources)
        self.k = k
        self.preemption_points = preemption_points

    def __copy__(self):
        return PeriodicBudgetResourceTask(name=self.name, c=self.c, p=self.p, k=self.k,
                                          preemption_points=self.preemption_points, resources=self.resources,
                                          locked_resources=self.locked_resources)


class DAGSubTask(Task):
    def __init__(self, name, c, a=0, d=None, parents=None, children=None, dist=0):
        """
        A Task class that represents a sub-task of a DAGTask. Specifies precedence relations with parents and children
        arguments. dist can be used to specify a distance from the sink node in the DAG.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete task
        :param a: type int
            The time unit at which the task becomes available
        :param d: type int
            The time unit at which the task is due
        :param parents: type list
            A list of DAGSubTask objects that this DAGSubTask depends on
        :param children: type list
            A list of DAGSubTask objects that depend on this DAGSubTask
        :param dist: type int
            Amount of distance from the sink node of the DAG
        """
        super(DAGSubTask, self).__init__(name=name, c=c, a=a, d=d)
        if parents is None:
            parents = []
        self.parents = parents
        if children is None:
            children = []
        self.children = children
        self.dist = dist

    def add_parent(self, task):
        """
        Adds a parent to the internal set of parents
        :param task: type DAGSubTask
            A DAGSubTask that this DAGSubTask depends on
        :return: None
        """
        self.parents = list(set(self.parents + [task]))

    def add_child(self, task):
        """
        Adds a child to the internal set of children
        :param task: type DAGSubTask
            A DAGSubTask that depends on this DAGSubTask
        :return: None
        """
        self.children = list(set(self.children + [task]))

    def __copy__(self):
        return DAGSubTask(name=self.name, c=self.c, d=self.d, dist=self.dist)


class DAGResourceSubTask(ResourceTask):
    def __init__(self, name, c=1, a=0, d=None, parents=None, children=None, resources=None, locked_resources=None,
                 dist=0):
        """
        A combination of a DAGSubTask and a ResourceTask.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete task
        :param a: type int
            The time unit at which the task becomes available
        :param d: type int
            The time unit at which the task is due
        :param parents: type list
            A list of DAGSubTask objects that this DAGSubTask depends on
        :param children: type list
            A list of DAGSubTask objects that depend on this DAGSubTask
        :param resources: type list
            List of resource identifiers (any object)
        :param locked_resources: type list
            List of resource identifiers (any object)
        :param dist: type int
            Amount of distance from the sink node of the DAG
        """
        super(DAGResourceSubTask, self).__init__(name, a=a, c=c, d=d, resources=resources,
                                                 locked_resources=locked_resources)
        self.d = d
        if parents is None:
            parents = []
        self.parents = parents
        if children is None:
            children = []
        self.children = children
        self.dist = dist

    def add_parent(self, task):
        """
        Adds a parent to the internal set of parents
        :param task: type DAGSubTask
            A DAGSubTask that this DAGSubTask depends on
        :return: None
        """
        self.parents = list(set(self.parents + [task]))

    def add_child(self, task):
        """
        Adds a child to the internal set of children
        :param task: type DAGSubTask
            A DAGSubTask that depends on this DAGSubTask
        :return: None
        """
        self.children = list(set(self.children + [task]))

    def __copy__(self):
        return DAGResourceSubTask(name=self.name, c=self.c, a=self.a, d=self.d, resources=self.resources,
                                  dist=self.dist, locked_resources=self.locked_resources)


class DAGBudgetResourceSubTask(DAGResourceSubTask):
    def __init__(self, name, c=1, a=0, d=None, k=0, parents=None, children=None, resources=None, locked_resources=None,
                 dist=0):
        """
        A combination of a DAGSubTask, ResourceTask, and BudgetTask.
        :param name: type str
            Name of the task
        :param c: type int
            Number of time units needed to complete task
        :param a: type int
            The time unit at which the task becomes available
        :param d: type int
            The time unit at which the task is due
        :param k: type int
            The number of time units the task may spend preempted
        :param parents: type list
            A list of DAGSubTask objects that this DAGSubTask depends on
        :param children: type list
            A list of DAGSubTask objects that depend on this DAGSubTask
        :param resources: type list
            List of resource identifiers (any object)
        :param locked_resources: type list
            List of resource identifiers (any object)
        :param dist: type int
            Amount of distance from the sink node of the DAG
        """
        super(DAGBudgetResourceSubTask, self).__init__(name=name, c=c, a=a, d=d, parents=parents, children=children,
                                                       resources=resources, locked_resources=locked_resources,
                                                       dist=dist)
        self.k = k

    def __copy__(self):
        return DAGBudgetResourceSubTask(name=self.name, c=self.c, a=self.a, d=self.d, k=self.k,
                                        resources=self.resources, locked_resources=self.locked_resources,
                                        dist=self.dist)


class DAGTask(Task):
    def __init__(self, name, tasks, d=None):
        """
        A DAGTask that encodes a set of subtasks with precedence relations.
        :param name: type str
            The name of the DAGTask
        :param tasks: type list
            List of DAGSubTasks that make up the precedence relation DAG
        :param d: type int
            The deadline that the set of sinks must complete before.
        """
        self.sources = []
        self.sinks = []
        self.tasks = {}
        self.subtasks = tasks

        # Get all the sources and sinks of the DAGTask
        for task in tasks:
            if not task.parents:
                self.sources.append(task)
            if not task.children:
                self.sinks.append(task)
            self.tasks[task.name] = task

        c = get_dag_exec_time(self)

        if d is None:
            if not all([t.d is None for t in self.sinks]):
                d = max([t.d for t in self.sinks])

        super(DAGTask, self).__init__(name=name, c=c, d=d)

    def __copy__(self):
        tasks = {}
        q = [t for t in self.sources]

        # Copy each subtask and reconstruct parent/child
        while q:
            original_task = q.pop(0)
            task = tasks.get(original_task.name, copy(original_task))
            tasks[task.name] = task

            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

            for original_child_task in original_task.children:
                if original_child_task not in q:
                    q.append(original_child_task)

        return DAGTask(name=self.name, tasks=list(tasks.values()), d=self.d)


class PeriodicDAGTask(PeriodicTask):
    def __init__(self, name, tasks, p):
        """
        A combination of the PeriodicTask and DAGTask.
        :param name: type str
            The name of the PeriodicDAGTask
        :param tasks: type list
            List of DAGSubTasks used to initialize DAGTask instances
        :param p: type int
            The number of time units between periodic releases of the task
        """
        self.sources = []
        self.sinks = []
        self.tasks = {}
        self.subtasks = tasks
        for task in tasks:
            if not task.parents:
                self.sources.append(task)
            if not task.children:
                self.sinks.append(task)
            self.tasks[task.name] = task

        c = get_dag_exec_time(self)

        super(PeriodicDAGTask, self).__init__(name=name, c=c, p=p)

    def __copy__(self):
        tasks = {}
        q = [t for t in self.sources]

        # Copy tasks and reconstruct parent/child relationship
        while q:
            original_task = q.pop(0)
            task = tasks.get(original_task.name, copy(original_task))
            tasks[task.name] = task

            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

            for original_child_task in original_task.children:
                if original_child_task not in q:
                    q.append(original_child_task)

        return PeriodicDAGTask(name=self.name, tasks=list(tasks.values()), p=self.p)


class ResourceDAGTask(ResourceTask):
    def __init__(self, name, tasks, a=0, d=None):
        """
        Combination of a DAGTask and a ResourceTask that tracks the resources needed for the entire DAG
        :param name: type str
            The name of the DAGTask
        :param tasks: type list
            List of DAGSubTasks that make up the precedence relation DAG
        :param a: type int
            The time unit at which the task becomes available
        :param d: type int
            The deadline that the set of sinks must complete before.
        """
        self.sources = []
        self.sinks = []
        self.tasks = {}
        self.subtasks = tasks
        for task in tasks:
            if not task.parents or all([p not in tasks for p in task.parents]):
                self.sources.append(task)
            if not task.children or all([c not in tasks for c in task.children]):
                self.sinks.append(task)
            self.tasks[task.name] = task

        c = get_dag_exec_time(self)
        if d is None:
            if not all([t.d is None for t in self.sinks]):
                d = max([t.d for t in self.sinks])

        super(ResourceDAGTask, self).__init__(name=name, a=a, c=c, d=d)
        self.assign_subtask_deadlines()

        self.resources = set()
        for task in tasks:
            self.resources |= set(task.resources)
            if task.locked_resources:
                self.resources |= set(task.locked_resources)

    def assign_subtask_deadlines(self):
        """
        Assigns deadlines to the DAGSubTasks composing the DAGTask
        :return: None
        """
        q = [t for t in self.sinks]
        processed = defaultdict(int)
        while q:
            task = q.pop(0)
            if task in self.sinks:
                task.d = self.d
            else:
                task.d = min([ct.d - ct.c for ct in task.children])
            processed[task.name] = 1
            for t in task.parents:
                if processed[t.name] == 0:
                    q.append(t)

    def get_resource_schedules(self):
        """
        Obtains the occupation schedules of the resources used by the DAGTask
        :return: type dict
            A dictionary of reosurce identifiers to lists of occupied time slots by the DAGTask's sub tasks
        """
        # Create schedules for each resource
        full_resource_schedules = defaultdict(list)
        for task in self.subtasks:
            resource_schedules = task.get_resource_schedules()
            for resource, slots in resource_schedules.items():
                full_resource_schedules[resource] += slots

        # Sort the slots in each schedule, record where occupation tasks are in schedule
        for resource in full_resource_schedules.keys():
            full_resource_schedules[resource] = list(sorted(full_resource_schedules[resource], key=lambda s: s[0]))
            additional_slots = []

            # Add occupation slots between tasks
            for (s1, t1), (s2, t2) in zip(full_resource_schedules[resource], full_resource_schedules[resource][1:]):
                if t1 != t2 and s2 - s1 > 1 and resource in t1.locked_resources:
                    occ_task = ResourceTask(name="Occupation", c=s2 - s1, a=s1 + 1, resources=[resource],
                                            locked_resources=[resource])
                    additional_slots += [(s1 + i, occ_task) for i in range(1, s2 - s1)]

            # Add occupation slots between last tasks and end of DAGTask
            last_slot, last_task = full_resource_schedules[resource][-1]
            completion_time = self.c + self.a
            if last_task.locked_resources and resource in last_task.locked_resources and \
                    completion_time - last_slot > 0:
                occ_task = ResourceTask(name="Occupation", c=completion_time - last_slot, a=last_slot + 1,
                                        resources=[resource], locked_resources=[resource])
                additional_slots += [(last_slot + i, occ_task) for i in range(1, completion_time - last_slot)]

            full_resource_schedules[resource] += additional_slots
            full_resource_schedules[resource] = list(sorted(full_resource_schedules[resource], key=lambda s: s[0]))

        return dict(full_resource_schedules)

    def get_resource_intervals(self, separate_occupation=False):
        """
        Obtains the intervaltrees representing the occupation of the resources used by the DAGTask
        :param separate_occupation: type bool
            Toggles whether occupation activities are split into separate tasks each time unit or one continuous
            one
        :return: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing the periods of occupation
        """
        # Take the resource schedules
        resource_schedules = self.get_resource_schedules()

        # Take each resource and convert slots to IntervalTree
        resource_intervals = defaultdict(IntervalTree)
        for resource, schedule in resource_schedules.items():
            s, t = schedule[0]
            interval = Interval(begin=s, end=s + 1, data=t)

            # If consecutive slots belong to same task extend the interval, otherwise
            # add it to the IntervalTree and start a new one
            for s, t in schedule[1:]:
                if t == interval.data and (not separate_occupation or t.name[0] != "O"):
                    interval = Interval(begin=interval.begin, end=s + 1, data=t)
                else:
                    resource_intervals[resource].add(interval)
                    interval = Interval(begin=s, end=s + 1, data=t)

            resource_intervals[resource].add(interval)

        return resource_intervals

    def __copy__(self):
        tasks = {}
        q = [t for t in self.sources]
        while q:
            original_task = q.pop(0)
            task = tasks.get(original_task.name, copy(original_task))
            tasks[task.name] = task

            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

            for original_child_task in original_task.children:
                if original_child_task not in q:
                    q.append(original_child_task)

        return ResourceDAGTask(name=self.name, tasks=list(tasks.values()), a=self.a, d=self.d)


class BudgetResourceDAGTask(ResourceDAGTask):
    def __init__(self, name, tasks, a=0, d=None, k=0):
        """
        Combination of a DAGTask, BudgetTask, and ResourceTask
        :param name: type str
            Name of the task
        :param tasks: type list
            List of DAGSubTasks that compose the DAG
        :param a: type int
            The release time of the DAGTask
        :param d: type int
            The deadline of the DAGTask
        :param k: type int
            The preemption budget of the DAGTask
        """
        super(BudgetResourceDAGTask, self).__init__(name=name, tasks=tasks, a=a, d=d)
        self.k = k

    def __copy__(self):
        tasks = {}
        q = [t for t in self.sources]
        while q:
            original_task = q.pop(0)
            task = tasks.get(original_task.name, copy(original_task))
            tasks[task.name] = task

            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

            for original_child_task in original_task.children:
                if original_child_task not in q:
                    q.append(original_child_task)

        return BudgetResourceDAGTask(name=self.name, tasks=list(tasks.values()), a=self.a, d=self.d, k=self.k)


class PeriodicResourceDAGTask(PeriodicDAGTask):
    def __init__(self, name, tasks, p):
        """
        A combination of a PeriodicDAGTask and a ResourceDAGTask
        :param name: type str
            Name of the dag task
        :param tasks: type list
            List of DAGSubTasks that represent the DAG of the task
        :param p: type int
            The period at which the task is released into the system
        """
        super(PeriodicResourceDAGTask, self).__init__(name=name, tasks=tasks, p=p)

        self.resources = set()
        for task in tasks:
            self.resources |= set(task.resources)
            if task.locked_resources:
                self.resources |= set(task.locked_resources)

    def get_resources(self):
        """
        Returns the tasks resources
        :return: type set
            A set of the resources used
        """
        return self.resources

    def get_resource_schedules(self):
        """
        Obtains the occupation schedules of the resources used by the DAGTask
        :return: type dict
            A dictionary of resource identifiers to lists of occupied time slots by the DAGTask's sub tasks
        """
        # Get the slots for each resource
        full_resource_schedules = defaultdict(list)
        for task in self.subtasks:
            resource_schedules = task.get_resource_schedules()
            for resource, slots in resource_schedules.items():
                full_resource_schedules[resource] += slots

        # Add occupation tasks between tasks on resource
        for resource in full_resource_schedules.keys():
            full_resource_schedules[resource] = list(sorted(full_resource_schedules[resource], key=lambda s: s[0]))
            additional_slots = []
            for (s1, t1), (s2, t2) in zip(full_resource_schedules[resource], full_resource_schedules[resource][1:]):
                if t1 != t2 and s2 - s1 > 1 and resource in t1.locked_resources:
                    occ_task = ResourceTask(name="Occupation", c=s2 - s1, a=s1 + 1, resources=[resource],
                                            locked_resources=[resource])
                    additional_slots += [(s1 + i, occ_task) for i in range(1, s2 - s1)]

            # Add occupation slots from last task to end of DAGTask
            last_slot, last_task = full_resource_schedules[resource][-1]
            completion_time = self.c + self.a
            if last_task.locked_resources and resource in last_task.locked_resources and \
                    completion_time - last_slot > 0:
                occ_task = ResourceTask(name="Occupation", c=completion_time - last_slot, a=last_slot + 1,
                                        resources=[resource], locked_resources=[resource])
                additional_slots += [(last_slot + i, occ_task) for i in range(1, completion_time - last_slot)]

            full_resource_schedules[resource] += additional_slots
            full_resource_schedules[resource] = list(sorted(full_resource_schedules[resource], key=lambda s: s[0]))

        return dict(full_resource_schedules)

    def get_resource_intervals(self, separate_occupation=False):
        """
        Obtains the intervaltrees representing the occupation of the resources used by the DAGTask
        :param separate_occupation: type bool
            Toggles whether occupation activities are split into separate tasks each time unit or one continuous
            one
        :return: type dict(IntervalTree)
            A dictionary of resource identifiers to interval trees representing the periods of occupation
        """
        # Get the resource schedules
        resource_schedules = self.get_resource_schedules()

        # Convert resource schedules to IntervalTree
        resource_intervals = defaultdict(IntervalTree)
        for resource, schedule in resource_schedules.items():
            s, t = schedule[0]
            interval = Interval(begin=s, end=s + 1, data=t)
            # Check consecutive slots to see if a new interval needs to be made
            for s, t in schedule[1:]:
                if t == interval.data and (not separate_occupation or t.name[0] != "O"):
                    interval = Interval(begin=interval.begin, end=s + 1, data=t)
                else:
                    resource_intervals[resource].add(interval)
                    interval = Interval(begin=s, end=s + 1, data=t)

            resource_intervals[resource].add(interval)

        return resource_intervals

    def __copy__(self):
        tasks = {}
        q = [t for t in self.sources]
        while q:
            original_task = q.pop(0)
            task = tasks.get(original_task.name, copy(original_task))
            tasks[task.name] = task

            for original_child_task in original_task.children:
                if original_child_task not in q:
                    q.append(original_child_task)

        for original_task in self.subtasks:
            task = tasks[original_task.name]
            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

        return PeriodicResourceDAGTask(name=self.name, tasks=list(tasks.values()), p=self.p)


class PeriodicBudgetResourceDAGTask(PeriodicResourceDAGTask):
    def __init__(self, name, tasks, p, k=0):
        """
        Combination of a PeriodicDAGTask and a BudgetResourceDAGTask
        :param name: type str
            Name of the dagtask
        :param tasks: type list
            List of DAGSubTasks that describe the DAG of the task
        :param p: type int
            The period at which the task is released
        :param k: type int
            The preemption budget of each instance of the DAGTask
        """
        super(PeriodicBudgetResourceDAGTask, self).__init__(name=name, tasks=tasks, p=p)
        self.k = k

    def __copy__(self):
        tasks = {}
        q = [t for t in self.sources]
        while q:
            original_task = q.pop(0)
            task = tasks.get(original_task.name, copy(original_task))
            tasks[task.name] = task

            for original_child_task in original_task.children:
                if original_child_task not in q:
                    q.append(original_child_task)

        for original_task in self.subtasks:
            task = tasks[original_task.name]
            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

        return PeriodicBudgetResourceDAGTask(name=self.name, tasks=list(tasks.values()), p=int(self.p), k=int(self.k))

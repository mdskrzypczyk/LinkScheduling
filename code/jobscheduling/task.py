from functools import reduce
from math import gcd, ceil
from jobscheduling.log import LSLogger
from collections import defaultdict
from copy import copy
from intervaltree import IntervalTree, Interval
import itertools

logger = LSLogger()


def lcm(a, b):
    return a * b // gcd(a, b)


def get_lcm_for(values):
    return reduce(lambda x, y: lcm(x, y), values)


def gcd_rationals(x, y):
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
    return reduce(lambda x, y: gcd_rationals(x, y), values)


def to_ranges(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def generate_non_periodic_task_set(periodic_task_set):
    periods = [task.p for task in periodic_task_set]
    schedule_length = get_lcm_for(periods)
    logger.debug("Computed hyperperiod {}".format(schedule_length))
    taskset = []
    for task in periodic_task_set:
        # Generate a task for each period the task executes
        num_tasks = schedule_length // task.p
        for i in range(num_tasks):
            taskset.append(Task(name="{}|{}".format(task.name, i), c=task.c, a=task.a + task.p * i,
                                d=task.a + task.p * (i + 1)))

    return taskset


def generate_non_periodic_dagtask_set(periodic_task_set):
    periods = [task.p for task in periodic_task_set]
    schedule_length = get_lcm_for(periods)
    logger.debug("Computed hyperperiod {}".format(schedule_length))
    taskset = []
    for task in periodic_task_set:
        # Generate a task for each period the task executes
        num_tasks = schedule_length // task.p
        for i in range(num_tasks):
            dag_copy = copy(task)
            release_offset = dag_copy.a + dag_copy.p * i
            for subtask in dag_copy.subtasks:
                subtask.a += release_offset
            new_task = ResourceDAGTask(name="{}|{}".format(dag_copy.name, i), a=release_offset,
                                       d=dag_copy.a + dag_copy.p * (i + 1), tasks=dag_copy.subtasks)
            taskset.append(new_task)

    return taskset


def generate_non_periodic_budget_task_set(periodic_task_set):
    periods = [task.p for task in periodic_task_set]
    schedule_length = get_lcm_for(periods)
    logger.debug("Computed hyperperiod {}".format(schedule_length))
    taskset = []
    for task in periodic_task_set:
        # Generate a task for each period the task executes
        num_tasks = schedule_length // task.p
        for i in range(num_tasks):
            taskset.append(
                BudgetTask(name="{}|{}".format(task.name, i), c=task.c, a=task.a + task.p * i,
                           d=task.a + task.p * (i + 1), k=task.k))

    return taskset


def find_dag_task_preemption_points(budget_dag_task, resources=None):
    if resources is None:
        resources = budget_dag_task.resources

    resource_intervals = budget_dag_task.get_resource_intervals(separate_occupation=True)
    global_itree = IntervalTree()
    for resource in resources:
        global_itree |= resource_intervals[resource]

    global_itree.merge_overlaps()
    preemption_points = list(sorted([(interval.begin, interval.end) for interval in global_itree
                                     if interval.end <= budget_dag_task.a + budget_dag_task.c]))
    points_to_locked_resources = defaultdict(list)
    points_to_needed_resources = defaultdict(list)

    points_to_subtasks = defaultdict(set)
    for point_start, point_end in preemption_points:
        for resource, itree in resource_intervals.items():
            locking_intervals = sorted(itree[0:point_end])
            if locking_intervals and point_end < budget_dag_task.a + budget_dag_task.c:
                last_task = locking_intervals[-1].data
                if last_task.locked_resources and resource in last_task.locked_resources:
                    points_to_locked_resources[(point_start, point_end)].append(resource)
            active_intervals = sorted(itree[point_start:budget_dag_task.a + budget_dag_task.c])
            if active_intervals:
                points_to_needed_resources[(point_start, point_end)].append(resource)
                active_intervals = sorted(itree[point_start:point_end])
                for interval in active_intervals:
                    if type(interval.data) != ResourceTask:
                        points_to_subtasks[(point_start, point_end)] |= {interval.data}

    preemption_points = [(point, points_to_locked_resources[point], points_to_needed_resources[point],
                          points_to_subtasks[point]) for point in preemption_points]
    all_pp_subtasks = list()
    for taskset in points_to_subtasks.values():
        all_pp_subtasks += list(filter(lambda task: type(task) != ResourceTask, taskset))

    if len(all_pp_subtasks) != len(budget_dag_task.subtasks) or \
            any([t not in all_pp_subtasks for t in budget_dag_task.subtasks]):
        print("Incorrect subtask set")
        import pdb
        pdb.set_trace()

    return preemption_points


class Task:
    def __init__(self, name, c, a=0, d=None, description=None, **kwargs):
        self.name = name
        self.a = a
        self.c = c
        self.d = d
        self.description = description

    def __lt__(self, other):
        return self.name < other.name

    def __copy__(self):
        return Task(name=self.name, a=self.a, c=self.c, d=self.d)


class BudgetTask(Task):
    def __init__(self, name, c, a=0, d=None, k=0, preemption_points=None):
        super(BudgetTask, self).__init__(name=name, c=c, a=a, d=d)
        self.k = k
        self.preemption_points = preemption_points

    def __copy__(self):
        return BudgetTask(name=self.name, c=self.c, a=self.a, d=self.d, k=self.k,
                          preemption_points=self.preemption_points)


class ResourceTask(Task):
    def __init__(self, name, c, a=0, d=None, resources=None, locked_resources=None):
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
        resource_schedules = defaultdict(list)
        slots = [(self.a + i, self) for i in range(ceil(self.c))]
        for resource in self.resources:
            resource_schedules[resource] += slots
        return dict(resource_schedules)

    def get_resource_intervals(self):
        resource_intervals = defaultdict(IntervalTree)
        if self.c > 0:
            interval = Interval(self.a, self.a + self.c)
            for resource in self.resources:
                resource_intervals[resource].add(interval)
        return resource_intervals

    def remap_resources(self, resource_relations):
        new_resources = []
        for resource in self.resources:
            new_resources.append(resource_relations[resource])

        new_locked_resources = []
        for resource in self.locked_resources:
            new_locked_resources.append(resource_relations[resource])

        self.resources = new_resources
        self.locked_resources = new_locked_resources


class BudgetResourceTask(ResourceTask):
    def __init__(self, name, c, a=0, d=None, k=0, preemption_points=None, resources=None, locked_resources=None):
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
        super(PeriodicTask, self).__init__(name=name, a=a, c=c)
        self.p = p

    def __copy__(self):
        return PeriodicTask(name=self.name, a=self.a, c=self.c, p=self.p)


class PeriodicBudgetTask(BudgetTask):
    def __init__(self, name, c, a=0, p=None, k=0, preemption_points=None):
        super(PeriodicBudgetTask, self).__init__(name=name, a=a, c=c, k=k, preemption_points=preemption_points)
        self.p = p

    def __copy__(self):
        return PeriodicBudgetTask(name=self.name, c=self.c, a=self.a, p=self.p, k=self.k,
                                  preemption_points=self.preemption_points)


class PeriodicResourceTask(ResourceTask):
    def __init__(self, name, c, p=None, resources=None, locked_resources=None):
        super(PeriodicResourceTask, self).__init__(name=name, c=c, resources=resources,
                                                   locked_resources=locked_resources)
        self.p = p

    def __copy__(self):
        return PeriodicResourceTask(name=self.name, c=self.c, resources=self.resources,
                                    locked_resources=self.locked_resources)


class PeriodicBudgetResourceTask(PeriodicResourceTask):
    def __init__(self, name, c, p=None, k=0, preemption_points=None, resources=None, locked_resources=None):
        super(PeriodicBudgetResourceTask, self).__init__(name=name, c=c, p=p, resources=resources,
                                                         locked_resources=locked_resources)
        self.k = k
        self.preemption_points = preemption_points

    def __copy__(self):
        return PeriodicBudgetResourceTask(name=self.name, c=self.c, p=self.p, k=self.k,
                                          preemption_points=self.preemption_points, resources=self.resources,
                                          locked_resources=self.locked_resources)


class DAGSubTask(Task):
    def __init__(self, name, c, d=None, parents=None, children=None, dist=0):
        super(DAGSubTask, self).__init__(name, c=c, d=None)
        self.d = d
        self.parents = parents
        self.children = children
        self.dist = dist

    def __copy__(self):
        return DAGSubTask(name=self.name, c=self.c, d=self.d, dist=self.dist)


class DAGResourceSubTask(ResourceTask):
    def __init__(self, name, c=1, a=0, d=None, parents=None, children=None, resources=None, locked_resources=None,
                 dist=0):
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
        self.parents = list(set(self.parents + [task]))

    def add_child(self, task):
        self.children = list(set(self.children + [task]))

    def __copy__(self):
        return DAGResourceSubTask(name=self.name, c=self.c, a=self.a, d=self.d, resources=self.resources,
                                  dist=self.dist, locked_resources=self.locked_resources)


class DAGBudgetResourceSubTask(DAGResourceSubTask):
    def __init__(self, name, c=1, a=0, d=None, k=0, parents=None, children=None, resources=None, locked_resources=None,
                 dist=0):
        super(DAGBudgetResourceSubTask, self).__init__(name=name, c=c, a=a, d=d, parents=parents, children=children,
                                                       resources=resources, locked_resources=locked_resources,
                                                       dist=dist)
        self.k = k

    def __copy__(self):
        return DAGBudgetResourceSubTask(name=self.name, c=self.c, a=self.a, d=self.d, k=self.k, parents=self.parents,
                                        children=self.children, resources=self.resources,
                                        locked_resources=self.locked_resources, dist=self.dist)


def get_dag_exec_time(dag):
    if not dag.subtasks:
        import pdb
        pdb.set_trace()
    return max([subtask.a + subtask.c for subtask in dag.subtasks]) - min([source.a for source in dag.sources])


class DAGTask(Task):
    def __init__(self, name, tasks, d=None):
        self.sources = []
        self.sinks = []
        self.tasks = {}
        for task in tasks:
            if task.parents is None:
                self.sources.append(task)
            if task.children is None:
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
        while q:
            original_task = q.pop(0)
            task = tasks.get(original_task.name, copy(original_task))
            tasks[task.name] = task

            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

            for original_child_task in original_task.children:
                q.append(original_child_task)

        return DAGTask(name=self.name, tasks=list(tasks.values()), d=self.d)


class PeriodicDAGTask(PeriodicTask):
    def __init__(self, name, tasks, p):
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
        while q:
            original_task = q.pop(0)
            task = tasks.get(original_task.name, copy(original_task))
            tasks[task.name] = task

            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

            for original_child_task in original_task.children:
                q.append(original_child_task)

        return PeriodicDAGTask(name=self.name, tasks=list(tasks.values()), p=self.p)


class ResourceDAGTask(ResourceTask):
    def __init__(self, name, tasks, a=0, d=None):
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

        if any([t.d > self.d for t in self.subtasks]):
            import pdb
            pdb.set_trace()

    def earliest_deadline(self):
        return min([subtask.d for subtask in self.sources])

    def get_resource_schedules(self):
        full_resource_schedules = defaultdict(list)
        for task in self.subtasks:
            resource_schedules = task.get_resource_schedules()
            for resource, slots in resource_schedules.items():
                full_resource_schedules[resource] += slots

        for resource in full_resource_schedules.keys():
            full_resource_schedules[resource] = list(sorted(full_resource_schedules[resource], key=lambda s: s[0]))
            additional_slots = []
            for (s1, t1), (s2, t2) in zip(full_resource_schedules[resource], full_resource_schedules[resource][1:]):
                if t1 != t2 and s2 - s1 > 1 and resource in t1.locked_resources:
                    occ_task = ResourceTask(name="Occupation", c=s2 - s1, a=s1 + 1, resources=[resource],
                                            locked_resources=[resource])
                    additional_slots += [(s1 + i, occ_task) for i in range(1, s2 - s1)]

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
        resource_schedules = self.get_resource_schedules()
        resource_intervals = defaultdict(IntervalTree)
        for resource, schedule in resource_schedules.items():
            s, t = schedule[0]
            interval = Interval(begin=s, end=s + 1, data=t)
            for s, t in schedule[1:]:
                if t == interval.data and (not separate_occupation or t.name[0] != "O"):
                    interval = Interval(begin=interval.begin, end=s + 1, data=t)
                else:
                    if resource_intervals[resource].overlap(interval.begin, interval.end):
                        import pdb
                        pdb.set_trace()
                    resource_intervals[resource].add(interval)
                    interval = Interval(begin=s, end=s + 1, data=t)

            if resource_intervals[resource].overlap(interval.begin, interval.end):
                import pdb
                pdb.set_trace()

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
                q.append(original_child_task)

        return ResourceDAGTask(name=self.name, tasks=list(tasks.values()), a=self.a, d=self.d)


class BudgetResourceDAGTask(ResourceDAGTask):
    def __init__(self, name, tasks, a=0, d=None, k=0):
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
                q.append(original_child_task)

        return BudgetResourceDAGTask(name=self.name, tasks=list(tasks.values()), a=self.a, d=self.d, k=self.k)


class PeriodicResourceDAGTask(PeriodicDAGTask):
    def __init__(self, name, tasks, p):
        super(PeriodicResourceDAGTask, self).__init__(name=name, tasks=tasks, p=p)

        self.resources = set()
        for task in tasks:
            self.resources |= set(task.resources)
            if task.locked_resources:
                self.resources |= set(task.locked_resources)

    def get_resources(self):
        return self.resources

    def get_resource_schedules(self):
        full_resource_schedules = defaultdict(list)
        for task in self.subtasks:
            resource_schedules = task.get_resource_schedules()
            for resource, slots in resource_schedules.items():
                full_resource_schedules[resource] += slots

        for resource in full_resource_schedules.keys():
            full_resource_schedules[resource] = list(sorted(full_resource_schedules[resource], key=lambda s: s[0]))
            additional_slots = []
            for (s1, t1), (s2, t2) in zip(full_resource_schedules[resource], full_resource_schedules[resource][1:]):
                if t1 != t2 and s2 - s1 > 1 and resource in t1.locked_resources:
                    occ_task = ResourceTask(name="Occupation", c=s2 - s1, a=s1 + 1, resources=[resource],
                                            locked_resources=[resource])
                    additional_slots += [(s1 + i, occ_task) for i in range(1, s2 - s1)]

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
        resource_schedules = self.get_resource_schedules()
        resource_intervals = defaultdict(IntervalTree)
        for resource, schedule in resource_schedules.items():
            s, t = schedule[0]
            interval = Interval(begin=s, end=s + 1, data=t)
            for s, t in schedule[1:]:
                if t == interval.data and (not separate_occupation or t.name[0] != "O"):
                    interval = Interval(begin=interval.begin, end=s + 1, data=t)
                else:
                    if resource_intervals[resource].overlap(interval.begin, interval.end):
                        import pdb
                        pdb.set_trace()
                    resource_intervals[resource].add(interval)
                    interval = Interval(begin=s, end=s + 1, data=t)

            if resource_intervals[resource].overlap(interval.begin, interval.end):
                import pdb
                pdb.set_trace()

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
                q.append(original_child_task)

        for original_task in self.subtasks:
            task = tasks[original_task.name]
            for original_parent_task in original_task.parents:
                parent_task = tasks[original_parent_task.name]
                parent_task.add_child(task)
                task.add_parent(parent_task)

        return PeriodicBudgetResourceDAGTask(name=self.name, tasks=list(tasks.values()), p=int(self.p), k=int(self.k))

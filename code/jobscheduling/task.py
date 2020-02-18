from functools import reduce    # need this line if you're using Python3.x


def lcm(a, b):
    if a > b:
        greater = a
    else:
        greater = b

    while True:
        if greater % a == 0 and greater % b == 0:
            lcm = greater
            break
        greater += 1

    return lcm


def get_lcm_for(values):
    return reduce(lambda x, y: lcm(x, y), values)


def generate_non_periodic_task_set(periodic_task_set):
    periods = [task.p for task in periodic_task_set]
    schedule_length = get_lcm_for(periods)
    taskset = []
    for task in periodic_task_set:
        # Generate a task for each period the task executes
        num_tasks = schedule_length // task.p
        for i in range(num_tasks):
            taskset.append(Task(name="{},{}".format(task.name, i), c=task.c, a=task.a + task.p*i, d=task.a + task.p * (i + 1)))

    return taskset


def generate_non_periodic_dagtask_set(periodic_task_set):
    periods = [task.p for task in periodic_task_set]
    schedule_length = get_lcm_for(periods)
    taskset = []
    for task in periodic_task_set:
        # Generate a task for each period the task executes
        num_tasks = schedule_length // task.p
        for i in range(num_tasks):
            taskset.append(
                ResourceDAGTask(name="{},{}".format(task.name, i), a=task.a + task.p * i, d=task.a + task.p * (i + 1), tasks=task.subtasks))

    return taskset


def generate_non_periodic_budget_task_set(periodic_task_set):
    periods = [task.p for task in periodic_task_set]
    schedule_length = get_lcm_for(periods)
    taskset = []
    for task in periodic_task_set:
        # Generate a task for each period the task executes
        num_tasks = schedule_length // task.p
        for i in range(num_tasks):
            taskset.append(
                BudgetTask(name="{},{}".format(task.name, i), c=task.c, a=task.a + task.p * i, d=task.a + task.p * (i + 1), k=task.k))

    return taskset


class Task:
    def __init__(self, name, c, a=0, d=None, **kwargs):
        self.name = name
        self.a = a
        self.c = c
        self.d = d

    def __lt__(self, other):
        return self.name < other.name


class BudgetTask(Task):
    def __init__(self, name, c, a=0, d=None, k=float('inf')):
        super(BudgetTask, self).__init__(name=name, c=c, a=a, d=d)
        self.k = k


class ResourceTask(Task):
    def __init__(self, name, c, a=0, d=None, resources=None):
        super(ResourceTask, self).__init__(name=name, c=c, a=a, d=d)
        self.resources = resources


class PeriodicTask(Task):
    def __init__(self, name, c, a=0, p=None):
        super(PeriodicTask, self).__init__(name=name, a=a, c=c)
        self.p = p


class PeriodicBudgetTask(BudgetTask):
    def __init__(self, name, c, a=0, p=None, k=0):
        super(PeriodicBudgetTask, self).__init__(name=name, a=a, c=c, k=k)
        self.p = p


class PeriodicResourceTask(ResourceTask):
    def __init__(self, name, c, p=None, resources=None):
        super(PeriodicResourceTask, self).__init__(name=name, c=c, resources=resources)
        self.p = p


class DAGSubTask(Task):
    def __init__(self, name, c, d=None, parents=None, children=None, dist=0):
        super(DAGSubTask, self).__init__(name, c=c, d=None)
        self.d = d
        self.parents = parents
        self.children = children
        self.dist = dist


class DAGResourceSubTask(ResourceTask):
    def __init__(self, name, c=1, a=0, d=None, parents=None, children=None, resources=None, dist=0):
        super(DAGResourceSubTask, self).__init__(name, a=a, c=c, d=d, resources=resources)
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


def get_dag_exec_time(sources):
    max_time = 0
    if sources is None:
        return max_time

    for source in sources:
        source_time = source.c
        max_child_time = get_dag_exec_time(source.children)
        if source_time + max_child_time > max_time:
            max_time = source_time + max_child_time

    return max_time


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

        c = get_dag_exec_time(self.sources)

        if d is None:
            if not all([t.d is None for t in self.sinks]):
                d = max([t.d for t in self.sinks])

        super(DAGTask, self).__init__(name=name, c=c, d=d)


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

        c = get_dag_exec_time(self.sources)

        super(PeriodicDAGTask, self).__init__(name=name, c=c, p=p)


class ResourceDAGTask(ResourceTask):
    def __init__(self, name, tasks, a=0, d=None):
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

        c = get_dag_exec_time(self.sources)
        if d is None:
            if not all([t.d is None for t in self.sinks]):
                d = max([t.d for t in self.sinks])
        super(ResourceDAGTask, self).__init__(name=name, a=a, c=c, d=d)

        self.resources = set()
        for task in tasks:
            self.resources |= set(task.resources)

    def earliest_deadline(self):
        return min([subtask.d for subtask in self.sources])


class PeriodicResourceDAGTask(PeriodicDAGTask):
    def __init__(self, name, tasks, p):
        super(PeriodicResourceDAGTask, self).__init__(name=name, tasks=tasks, p=p)

        self.resources = set()
        for task in tasks:
            self.resources |= set(task.resources)

    def get_resources(self):
        return self.resources
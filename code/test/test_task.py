import unittest
from jobscheduling.task import Task, BudgetTask, ResourceTask, BudgetResourceTask, PeriodicTask, PeriodicBudgetTask, \
    PeriodicResourceTask, PeriodicBudgetResourceTask, DAGSubTask, DAGResourceSubTask, DAGBudgetResourceSubTask, \
    DAGTask, PeriodicDAGTask, ResourceDAGTask, BudgetResourceDAGTask, PeriodicResourceDAGTask, \
    PeriodicBudgetResourceDAGTask


class TestTask(unittest.TestCase):
    def test_init(self):
        test_name = "ABC"
        test_proc = 100
        test_release = 50
        test_deadline = 200
        test_description = "test"

        task = Task(name=test_name, c=test_proc, a=test_release, d=test_deadline, description=test_description)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.description, test_description)


class TestResourceTask(unittest.TestCase):
    def test_init(self):
        test_name = "ABC"
        test_proc = 100
        test_release = 50
        test_deadline = 200
        test_resources = ['1', '2', '3']
        test_locked_resources = ['4', '5', '6']

        task = ResourceTask(name=test_name, c=test_proc, a=test_release, d=test_deadline, resources=test_resources,
                            locked_resources=test_locked_resources)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.resources, test_resources)
        self.assertEqual(task.locked_resources, test_locked_resources)

        task = ResourceTask(name=test_name, c=test_proc, a=test_release, d=test_deadline)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.resources, [])
        self.assertEqual(task.locked_resources, [])

    def test_get_resource_schedules(self):
        test_name = "ABC"
        test_proc = 3
        test_release = 50
        test_deadline = 200
        test_resources = ['1', '2']
        test_locked_resources = ['3']

        task = ResourceTask(name=test_name, c=test_proc, a=test_release, d=test_deadline, resources=test_resources,
                            locked_resources=test_locked_resources)

        resource_schedules = task.get_resource_schedules()

        slots = [(s, task) for s in range(test_release, test_release + test_proc)]
        expected_resource_schedules = {
            test_resources[0]: slots,
            test_resources[1]: slots
        }

        for resource in resource_schedules.keys():
            self.assertEqual(resource_schedules[resource], expected_resource_schedules[resource])

    def test_get_resource_intervals(self):
        test_name = "ABC"
        test_proc = 3
        test_release = 50
        test_deadline = 200
        test_resources = ['1', '2']
        test_locked_resources = ['3']

        task = ResourceTask(name=test_name, c=test_proc, a=test_release, d=test_deadline, resources=test_resources,
                            locked_resources=test_locked_resources)

        resource_intervals = task.get_resource_intervals()

        for resource, interval_tree in resource_intervals.items():
            self.assertTrue(resource in test_resources)
            self.assertEqual(len(interval_tree), 1)
            interval = sorted(interval_tree)[0]
            self.assertEqual(interval.begin, test_release)
            self.assertEqual(interval.end, test_release + test_proc)


class TestBudgetTask(unittest.TestCase):
    def test_init(self):
        test_name = "ABC"
        test_proc = 100
        test_release = 50
        test_deadline = 200
        test_budget = 5
        test_preemption_points = [1, 4, 7]

        task = BudgetTask(name=test_name, c=test_proc, a=test_release, d=test_deadline, k=test_budget,
                          preemption_points=test_preemption_points)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.k, test_budget)
        self.assertEqual(task.preemption_points, test_preemption_points)


class TestBudgetResourceTask(unittest.TestCase):
    def test_init(self):
        test_name = "ABC"
        test_proc = 100
        test_release = 50
        test_deadline = 200
        test_budget = 5
        test_preemption_points = [1, 4, 7]
        test_resources = ['1', '2', '3']
        test_locked_resources = ['4', '5', '6']

        task = BudgetResourceTask(name=test_name, c=test_proc, a=test_release, d=test_deadline, k=test_budget,
                                  resources=test_resources, locked_resources=test_locked_resources,
                                  preemption_points=test_preemption_points)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.k, test_budget)
        self.assertEqual(task.preemption_points, test_preemption_points)
        self.assertEqual(task.resources, test_resources)
        self.assertEqual(task.locked_resources, test_locked_resources)


class TestPeriodicTasks(unittest.TestCase):
    def test_init_periodic_task(self):
        test_name = "ABC"
        test_proc = 100
        test_release = 50
        test_period = 200

        task = PeriodicTask(name=test_name, c=test_proc, a=test_release, p=test_period)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.p, test_period)

    def test_init_periodic_budget_task(self):
        test_name = "ABC"
        test_proc = 100
        test_release = 50
        test_period = 200
        test_budget = 5
        test_preemption_points = [1, 4, 7]

        task = PeriodicBudgetTask(name=test_name, c=test_proc, a=test_release, p=test_period, k=test_budget,
                                  preemption_points=test_preemption_points)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.p, test_period)
        self.assertEqual(task.k, test_budget)
        self.assertEqual(task.preemption_points, test_preemption_points)

    def test_init_periodic_resource_task(self):
        test_name = "ABC"
        test_proc = 100
        test_release = 50
        test_period = 200
        test_resources = ['1', '2', '3']
        test_locked_resources = ['4', '5', '6']

        task = PeriodicResourceTask(name=test_name, c=test_proc, a=test_release, p=test_period,
                                    resources=test_resources, locked_resources=test_locked_resources)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.p, test_period)
        self.assertEqual(task.resources, test_resources)
        self.assertEqual(task.locked_resources, test_locked_resources)

        task = PeriodicResourceTask(name=test_name, c=test_proc, a=test_release, p=test_period)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.p, test_period)
        self.assertEqual(task.resources, [])
        self.assertEqual(task.locked_resources, [])

    def test_init_periodic_budget_resource_task(self):
        test_name = "ABC"
        test_proc = 100
        test_release = 50
        test_period = 200
        test_budget = 5
        test_preemption_points = [1, 4, 7]
        test_resources = ['1', '2', '3']
        test_locked_resources = ['4', '5', '6']

        task = PeriodicBudgetResourceTask(name=test_name, c=test_proc, a=test_release, p=test_period, k=test_budget,
                                          resources=test_resources, locked_resources=test_locked_resources,
                                          preemption_points=test_preemption_points)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc)
        self.assertEqual(task.a, test_release)
        self.assertEqual(task.p, test_period)
        self.assertEqual(task.k, test_budget)
        self.assertEqual(task.preemption_points, test_preemption_points)
        self.assertEqual(task.resources, test_resources)
        self.assertEqual(task.locked_resources, test_locked_resources)


class TestDagSubTasks(unittest.TestCase):
    def test_init_dag_subtask(self):
        test_name = "ABC"
        test_proc_time = 10
        test_deadline = 100
        test_dist = 2

        task = DAGSubTask(name=test_name, c=test_proc_time)
        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc_time)
        self.assertIsNone(task.d)
        self.assertEqual(task.children, [])
        self.assertEqual(task.parents, [])

        test_parents = [DAGSubTask(name="Parent", c=2)]
        test_children = [DAGSubTask(name="Child", c=3)]
        task = DAGSubTask(name=test_name, c=test_proc_time, d=test_deadline, parents=test_parents,
                          children=test_children, dist=test_dist)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc_time)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.children, test_children)
        self.assertEqual(task.parents, test_parents)

    def test_init_dag_resource_subtask(self):
        test_name = "ABC"
        test_proc_time = 10
        test_deadline = 100
        test_dist = 2

        task = DAGResourceSubTask(name=test_name, c=test_proc_time)
        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc_time)
        self.assertIsNone(task.d)
        self.assertEqual(task.children, [])
        self.assertEqual(task.parents, [])
        self.assertEqual(task.resources, [])
        self.assertEqual(task.locked_resources, [])

        test_parents = [DAGSubTask(name="Parent", c=2)]
        test_children = [DAGSubTask(name="Child", c=3)]
        test_resources = ['2']
        test_locked_resources = ['5']
        task = DAGResourceSubTask(name=test_name, c=test_proc_time, d=test_deadline, parents=test_parents,
                          children=test_children, dist=test_dist, resources=test_resources,
                          locked_resources=test_locked_resources)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc_time)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.children, test_children)
        self.assertEqual(task.parents, test_parents)
        self.assertEqual(task.resources, test_resources)
        self.assertEqual(task.locked_resources, test_locked_resources)

    def test_init_dag_budget_resource_subtask(self):
        test_name = "ABC"
        test_proc_time = 10
        test_deadline = 100
        test_dist = 2

        task = DAGBudgetResourceSubTask(name=test_name, c=test_proc_time)
        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc_time)
        self.assertIsNone(task.d)
        self.assertEqual(task.k, 0)
        self.assertEqual(task.children, [])
        self.assertEqual(task.parents, [])
        self.assertEqual(task.resources, [])
        self.assertEqual(task.locked_resources, [])

        test_parents = [DAGSubTask(name="Parent", c=2)]
        test_children = [DAGSubTask(name="Child", c=3)]
        test_resources = ['2']
        test_locked_resources = ['5']
        test_budget = 10
        task = DAGBudgetResourceSubTask(name=test_name, c=test_proc_time, d=test_deadline, k=test_budget,
                                        parents=test_parents, children=test_children, dist=test_dist,
                                        resources=test_resources, locked_resources=test_locked_resources)

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.c, test_proc_time)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.k, test_budget)
        self.assertEqual(task.children, test_children)
        self.assertEqual(task.parents, test_parents)
        self.assertEqual(task.resources, test_resources)
        self.assertEqual(task.locked_resources, test_locked_resources)


class TestDagTasks(unittest.TestCase):
    def test_init_dagtask(self):
        test_name = "ABC"
        test_subtask_proc = 3
        test_task = DAGSubTask(name="SubTask", c=test_subtask_proc)

        task = DAGTask(name=test_name, tasks=[test_task])

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task])
        self.assertEqual(task.subtasks, [test_task])
        self.assertEqual(task.tasks, {test_task.name: test_task})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_subtask_proc)
        self.assertIsNone(task.d)

        test_task2 = DAGSubTask(name="Subtask2", c=4, a=test_task.c, parents=[test_task])
        test_task.add_child(test_task2)
        task = DAGTask(name=test_name, tasks=[test_task, test_task2])

        test_deadline = 100
        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task2])
        self.assertEqual(task.subtasks, [test_task, test_task2])
        self.assertEqual(task.tasks, {test_task.name: test_task, test_task2.name: test_task2})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_task.c + test_task2.c)
        self.assertIsNone(task.d, test_deadline)

    def test_init_resource_dagtask(self):
        test_name = "ABC"
        test_subtask_proc = 3
        test_deadline = 100

        test_task = DAGResourceSubTask(name="SubTask", c=test_subtask_proc)

        task = ResourceDAGTask(name=test_name, d=test_deadline, tasks=[test_task])

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task])
        self.assertEqual(task.subtasks, [test_task])
        self.assertEqual(task.tasks, {test_task.name: test_task})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_subtask_proc)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.resources, set())

        test_resources1 = ['1', '2']
        test_task = DAGResourceSubTask(name="Subtask1", c=4, a=0, resources=test_resources1)
        test_resources2 = ['3']
        test_task2 = DAGResourceSubTask(name="Subtask2", c=4, a=test_task.c, parents=[test_task],
                                        resources=test_resources2)
        test_task.add_child(test_task2)
        task = ResourceDAGTask(name=test_name, d=test_deadline, tasks=[test_task, test_task2])

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task2])
        self.assertEqual(task.subtasks, [test_task, test_task2])
        self.assertEqual(task.tasks, {test_task.name: test_task, test_task2.name: test_task2})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_task.c + test_task2.c)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.resources, set(test_task.resources + test_task2.resources))

    def test_init_budget_resource_dagtask(self):
        test_name = "ABC"
        test_subtask_proc = 3
        test_deadline = 100

        test_task = DAGResourceSubTask(name="SubTask", c=test_subtask_proc)

        task = BudgetResourceDAGTask(name=test_name, d=test_deadline, tasks=[test_task])

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task])
        self.assertEqual(task.subtasks, [test_task])
        self.assertEqual(task.tasks, {test_task.name: test_task})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_subtask_proc)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.k, 0)
        self.assertEqual(task.resources, set())

        test_resources1 = ['1', '2']
        test_task = DAGResourceSubTask(name="Subtask1", c=4, a=0, resources=test_resources1)
        test_resources2 = ['3']
        test_task2 = DAGResourceSubTask(name="Subtask2", c=4, a=test_task.c, parents=[test_task],
                                        resources=test_resources2)
        test_task.add_child(test_task2)
        test_budget = 20
        task = BudgetResourceDAGTask(name=test_name, d=test_deadline, k=test_budget, tasks=[test_task, test_task2])

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task2])
        self.assertEqual(task.subtasks, [test_task, test_task2])
        self.assertEqual(task.tasks, {test_task.name: test_task, test_task2.name: test_task2})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_task.c + test_task2.c)
        self.assertEqual(task.d, test_deadline)
        self.assertEqual(task.k, test_budget)
        self.assertEqual(task.resources, set(test_task.resources + test_task2.resources))

    def test_init_periodic_dagtask(self):
        test_name = "ABC"
        test_period = 100
        test_subtask_proc = 3
        test_task = DAGSubTask(name="SubTask", c=test_subtask_proc)

        task = PeriodicDAGTask(name=test_name, p=test_period, tasks=[test_task])

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task])
        self.assertEqual(task.subtasks, [test_task])
        self.assertEqual(task.tasks, {test_task.name: test_task})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_subtask_proc)
        self.assertEqual(task.p, test_period)

        test_task2 = DAGSubTask(name="Subtask2", c=4, a=test_task.c, parents=[test_task])
        test_task.add_child(test_task2)
        task = PeriodicDAGTask(name=test_name, p=test_period, tasks=[test_task, test_task2])

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task2])
        self.assertEqual(task.subtasks, [test_task, test_task2])
        self.assertEqual(task.tasks, {test_task.name: test_task, test_task2.name: test_task2})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_task.c + test_task2.c)
        self.assertEqual(task.p, test_period)

    def test_init_periodic_resource_dagtask(self):
        test_name = "ABC"
        test_period = 100
        test_subtask_proc = 3
        test_task = DAGResourceSubTask(name="SubTask", c=test_subtask_proc)

        task = PeriodicResourceDAGTask(name=test_name, p=test_period, tasks=[test_task])

        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task])
        self.assertEqual(task.subtasks, [test_task])
        self.assertEqual(task.tasks, {test_task.name: test_task})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_subtask_proc)
        self.assertEqual(task.p, test_period)
        self.assertEqual(task.resources, set())

        test_resources1 = ['1', '2']
        test_task = DAGResourceSubTask(name="Subtask1", c=4, a=0, resources=test_resources1)
        test_resources2 = ['3']
        test_task2 = DAGResourceSubTask(name="Subtask2", c=4, a=test_task.c, parents=[test_task],
                                        resources=test_resources2)
        test_task.add_child(test_task2)

        task = PeriodicResourceDAGTask(name=test_name, p=test_period, tasks=[test_task, test_task2])
        self.assertEqual(task.name, test_name)
        self.assertEqual(task.sources, [test_task])
        self.assertEqual(task.sinks, [test_task2])
        self.assertEqual(task.subtasks, [test_task, test_task2])
        self.assertEqual(task.tasks, {test_task.name: test_task, test_task2.name: test_task2})
        self.assertEqual(task.a, 0)
        self.assertEqual(task.c, test_task.c + test_task2.c)
        self.assertEqual(task.p, test_period)
        self.assertEqual(task.resources, set(test_task.resources + test_task2.resources))

    def test_init_periodic_budget_resource_dagtask(self):
        pass



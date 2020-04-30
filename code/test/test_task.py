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
    def test_init(self):
        pass


class TestDagTasks(unittest.TestCase):
    def test_init_task(self):
        pass

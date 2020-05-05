import unittest
from jobscheduling.schedulers.scheduler import Scheduler, CommonScheduler
from jobscheduling.task import Task, DAGBudgetResourceSubTask, BudgetResourceDAGTask, PeriodicBudgetResourceDAGTask, \
    BudgetResourceTask, PeriodicBudgetResourceTask


class TestScheduleVerification(unittest.TestCase):
    def test_verify_schedule(self):
        pass

    def test_verify_budget_schedule(self):
        pass

    def test_verify_segmented_budget_schedule(self):
        pass


class TestScheduler(unittest.TestCase):
    def test_init(self):
        s = Scheduler()
        self.assertEqual(s.curr_time, 0)
        self.assertEqual(s.schedule, [])
        self.assertIsNone(s.taskset)

    def test_preprocess_taskset(self):
        test_taskset = [1]
        s = Scheduler()
        processed = s.preprocess_taskset(test_taskset)
        self.assertEqual(test_taskset, processed)

    def test_add_to_schedule(self):
        test_tasks = [Task(name="t1", c=4), Task(name="t2", c=3), Task(name="t3", c=10)]
        s = Scheduler()

        for task in test_tasks:
            s.add_to_schedule(task=task, duration=task.c)

        expected_schedule = [(0, 4, test_tasks[0]), (4, 7, test_tasks[1]), (7, 17, test_tasks[2])]
        self.assertEqual(s.schedule, expected_schedule)

    def test_create_new_task_instance(self):
        subtask1 = DAGBudgetResourceSubTask(name="subtask1", c=3, resources=['1'])
        subtask2 = DAGBudgetResourceSubTask(name="subtask2", a=subtask1.c, c=4, parents=[subtask1], resources=['2'])
        subtask1.add_child(subtask2)
        test_periodic_task = PeriodicBudgetResourceDAGTask(name="t1", tasks=[subtask1, subtask2], p=30, k=100)

        s = Scheduler()
        task_instance = s.create_new_task_instance(test_periodic_task, 0)
        self.assertEqual(type(task_instance), BudgetResourceTask)
        self.assertEqual(task_instance.a, 0)
        self.assertEqual(task_instance.c, test_periodic_task.c)
        self.assertEqual(task_instance.d, test_periodic_task.p)
        self.assertEqual(task_instance.k, test_periodic_task.k)
        self.assertEqual(task_instance.resources, test_periodic_task.resources)
        self.assertEqual(task_instance.locked_resources, [])

        task_instance = s.create_new_task_instance(test_periodic_task, 5)
        self.assertEqual(type(task_instance), BudgetResourceTask)
        self.assertEqual(task_instance.a, 5*test_periodic_task.p)
        self.assertEqual(task_instance.c, test_periodic_task.c)
        self.assertEqual(task_instance.d, 6*test_periodic_task.p)
        self.assertEqual(task_instance.k, test_periodic_task.k)
        self.assertEqual(task_instance.resources, test_periodic_task.resources)
        self.assertEqual(task_instance.locked_resources, [])

    def test_initialize_taskset(self):
        subtask1 = DAGBudgetResourceSubTask(name="subtask1", c=3, resources=['1'])
        subtask2 = DAGBudgetResourceSubTask(name="subtask2", a=subtask1.c, c=4, parents=[subtask1], resources=['2'])
        subtask1.add_child(subtask2)
        test_periodic_task1 = PeriodicBudgetResourceDAGTask(name="t1", tasks=[subtask1, subtask2], p=30, k=100)

        subtask3 = DAGBudgetResourceSubTask(name="subtask3", c=10, resources=['4'])
        subtask4 = DAGBudgetResourceSubTask(name="subtask4", a=subtask3.c, c=2, parents=[subtask3], resources=['5'])
        subtask3.add_child(subtask4)
        test_periodic_task2 = PeriodicBudgetResourceDAGTask(name="t2", tasks=[subtask3, subtask4], p=15, k=20)

        s = Scheduler()
        t11, t21 = s.initialize_taskset([test_periodic_task1, test_periodic_task2])
        self.assertEqual(type(t11), BudgetResourceTask)
        self.assertEqual(t11.a, 0)
        self.assertEqual(t11.c, test_periodic_task1.c)
        self.assertEqual(t11.d, test_periodic_task1.p)
        self.assertEqual(t11.k, test_periodic_task1.k)
        self.assertEqual(t11.resources, test_periodic_task1.resources)
        self.assertEqual(t11.locked_resources, [])

        self.assertEqual(type(t21), BudgetResourceTask)
        self.assertEqual(t21.a, 0)
        self.assertEqual(t21.c, test_periodic_task2.c)
        self.assertEqual(t21.d, test_periodic_task2.p)
        self.assertEqual(t21.k, test_periodic_task2.k)
        self.assertEqual(t21.resources, test_periodic_task2.resources)
        self.assertEqual(t21.locked_resources, [])


class TestCommonScheduler(unittest.TestCase):
    def test_create_new_task_instance(self):
        subtask1 = DAGBudgetResourceSubTask(name="subtask1", c=3, resources=['1'])
        subtask2 = DAGBudgetResourceSubTask(name="subtask2", a=subtask1.c, c=4, parents=[subtask1], resources=['2'])
        subtask1.add_child(subtask2)
        test_periodic_task = PeriodicBudgetResourceDAGTask(name="t1", tasks=[subtask1, subtask2], p=30, k=100)

        s = CommonScheduler()
        task_instance = s.create_new_task_instance(test_periodic_task, 0)
        self.assertEqual(type(task_instance), BudgetResourceDAGTask)
        self.assertEqual(task_instance.a, 0)
        self.assertEqual(task_instance.c, test_periodic_task.c)
        self.assertEqual(task_instance.d, test_periodic_task.p)
        self.assertEqual(task_instance.k, test_periodic_task.k)
        self.assertEqual(len(task_instance.subtasks), len(test_periodic_task.subtasks))
        self.assertEqual(task_instance.resources, test_periodic_task.resources)
        self.assertEqual(task_instance.locked_resources, [])

        task_instance = s.create_new_task_instance(test_periodic_task, 5)
        self.assertEqual(type(task_instance), BudgetResourceDAGTask)
        self.assertEqual(task_instance.a, 5 * test_periodic_task.p)
        self.assertEqual(task_instance.c, test_periodic_task.c)
        self.assertEqual(task_instance.d, 6 * test_periodic_task.p)
        self.assertEqual(task_instance.k, test_periodic_task.k)
        self.assertEqual(len(task_instance.subtasks), len(test_periodic_task.subtasks))
        self.assertEqual(task_instance.resources, test_periodic_task.resources)
        self.assertEqual(task_instance.locked_resources, [])

    def test_remove_useless_resource_occupations(self):
        pass

import unittest
from jobscheduling.schedulers.BlockNPEDF import UniResourceBlockNPEDFScheduler, MultipleResourceBlockNPEDFScheduler
from jobscheduling.schedulers.BlockPBEDF import UniResourceConsiderateFixedPointPreemptionBudgetScheduler
from jobscheduling.task import DAGBudgetResourceSubTask, PeriodicBudgetResourceDAGTask


class TestUniResourceNPEDF(unittest.TestCase):
    def test_invalid_schedule(self):
        t1_subtask = DAGBudgetResourceSubTask(name="T1S", c=2, resources=['1'])
        t1 = PeriodicBudgetResourceDAGTask(name="T1", tasks=[t1_subtask], p=4)
        t2_subtask = DAGBudgetResourceSubTask(name="T2S", c=6, resources=['2'])
        t2 = PeriodicBudgetResourceDAGTask(name="T2", tasks=[t2_subtask], p=12)
        taskset = [t1, t2]

        scheduler = UniResourceBlockNPEDFScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(taskset)
        expected_schedule = [(0, 2, t1), (2, 8, t2), (8, 10, t1), (10, 12, t1)]

        self.assertEqual(tasks, taskset)
        self.assertFalse(valid)
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)

    def test_valid_schedule(self):
        t1_subtask = DAGBudgetResourceSubTask(name="T1S", c=2, resources=['1'])
        t1 = PeriodicBudgetResourceDAGTask(name="T1", tasks=[t1_subtask], p=4)
        t2_subtask = DAGBudgetResourceSubTask(name="T2S", c=2, resources=['2'])
        t2 = PeriodicBudgetResourceDAGTask(name="T2", tasks=[t2_subtask], p=4)
        taskset = [t1, t2]

        scheduler = UniResourceBlockNPEDFScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(taskset)
        expected_schedule = [(0, 2, t1), (2, 4, t2)]

        self.assertEqual(tasks, taskset)
        self.assertTrue(valid)
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)


class TestUniResourceCEDF(unittest.TestCase):
    def test_schedule(self):
        pass


class TestUniResourcePBEDF(unittest.TestCase):
    def test_schedule(self):
        pass


class TestUniResourceFixedPointPreemptionPBEDF(unittest.TestCase):
    def test_invalid_schedule(self):
        t1_subtask = DAGBudgetResourceSubTask(name="T1S", c=2, resources=['1'])
        t1 = PeriodicBudgetResourceDAGTask(name="T1", tasks=[t1_subtask], p=4)
        t2_subtask1 = DAGBudgetResourceSubTask(name="T2S1", c=2, resources=['2'])
        t2_subtask2 = DAGBudgetResourceSubTask(name="T2S2", a=t2_subtask1.c, c=4, resources=['2'],
                                               parents=[t2_subtask1])
        t2_subtask1.add_child(t2_subtask2)
        t2 = PeriodicBudgetResourceDAGTask(name="T2", tasks=[t2_subtask1, t2_subtask2], p=12)
        taskset = [t1, t2]

        scheduler = UniResourceConsiderateFixedPointPreemptionBudgetScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(taskset)

        self.assertEqual(tasks, taskset)
        self.assertFalse(valid)
        self.assertIsNone(schedule)

    def test_valid_schedule(self):
        t1_subtask = DAGBudgetResourceSubTask(name="T1S", c=2, resources=['1'])
        t1 = PeriodicBudgetResourceDAGTask(name="T1", tasks=[t1_subtask], p=4)
        t2_subtask1 = DAGBudgetResourceSubTask(name="T2S1", c=2, resources=['2'])
        t2_subtask2 = DAGBudgetResourceSubTask(name="T2S2", a=t2_subtask1.c, c=4, resources=['2'],
                                               parents=[t2_subtask1])
        t2_subtask1.add_child(t2_subtask2)
        t2 = PeriodicBudgetResourceDAGTask(name="T2", tasks=[t2_subtask1, t2_subtask2], p=12, k=2)
        taskset = [t1, t2]

        scheduler = UniResourceConsiderateFixedPointPreemptionBudgetScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(taskset)
        expected_schedule = [(0, 2, t1), (2, 4, t2), (4, 6, t1), (6, 10, t2), (10, 12, t1)]

        self.assertEqual(tasks, taskset)
        self.assertTrue(valid)
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)


class TestMultiResourceNPEDF(unittest.TestCase):
    def test_schedule(self):
        pass


class TestMultiResourceBlockNPEDF(unittest.TestCase):
    def test_schedule(self):
        pass


class TestMultiResourceCEDF(unittest.TestCase):
    def test_schedule(self):
        pass


class TestMultiResourcePBEDF(unittest.TestCase):
    def test_schedule(self):
        pass


class TestMultiResourceSegmentPBEDF(unittest.TestCase):
    def test_schedule(self):
        pass

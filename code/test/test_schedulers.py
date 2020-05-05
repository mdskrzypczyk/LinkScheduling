import unittest
from jobscheduling.protocols import to_ranges, convert_protocol_to_task, schedule_dag_for_resources
from jobscheduling.schedulers.BlockNPEDF import UniResourceBlockNPEDFScheduler, MultipleResourceBlockNPEDFScheduler
from jobscheduling.schedulers.BlockPBEDF import UniResourceConsiderateFixedPointPreemptionBudgetScheduler
from jobscheduling.schedulers.CEDF import UniResourceCEDFScheduler, MultipleResourceBlockCEDFScheduler
from jobscheduling.schedulers.NPEDF import MultipleResourceNonBlockNPEDFScheduler
from jobscheduling.schedulers.SearchBlockPBEDF import MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler,\
    MultipleResourceConsiderateSegmentPreemptionBudgetScheduler
from jobscheduling.task import DAGBudgetResourceSubTask, PeriodicBudgetResourceDAGTask
from jobscheduling.topology import gen_line_topology
from simulations.common import get_protocol_without_rate_constraint


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
    def test_valid_schedule_npedf(self):
        t1_subtask = DAGBudgetResourceSubTask(name="T1S", c=2, resources=['1'])
        t1 = PeriodicBudgetResourceDAGTask(name="T1", tasks=[t1_subtask], p=4)
        t2_subtask = DAGBudgetResourceSubTask(name="T2S", c=2, resources=['2'])
        t2 = PeriodicBudgetResourceDAGTask(name="T2", tasks=[t2_subtask], p=4)
        taskset = [t1, t2]

        scheduler = UniResourceCEDFScheduler()
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

    def test_valid_schedule(self):
        t1_subtask = DAGBudgetResourceSubTask(name="T1S", c=1, resources=['1'])
        t1 = PeriodicBudgetResourceDAGTask(name="T1", tasks=[t1_subtask], p=3)
        t2_subtask = DAGBudgetResourceSubTask(name="T2S", c=2, resources=['2'])
        t2 = PeriodicBudgetResourceDAGTask(name="T2", tasks=[t2_subtask], p=15)
        t3_subtask = DAGBudgetResourceSubTask(name="T3S", c=4, resources=['3'])
        t3 = PeriodicBudgetResourceDAGTask(name="T3", tasks=[t3_subtask], p=25)
        t4_subtask = DAGBudgetResourceSubTask(name="T4S", c=1, resources=['4'])
        t4 = PeriodicBudgetResourceDAGTask(name="T4", tasks=[t4_subtask], p=5)
        taskset = [t1, t2, t3, t4]

        scheduler = UniResourceBlockNPEDFScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(taskset)
        self.assertFalse(valid)

        scheduler = UniResourceCEDFScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(taskset)
        expected_occupied_time = [(0, 13), (15, 21), (24, 25), (27, 40), (42, 42), (45, 58), (60, 66), (69, 70),
                                  (72, 72)]
        occupied = []
        for s, e, t in schedule:
            occupied += [i for i in range(s, e)]

        ranges = list(to_ranges(occupied))
        self.assertEqual(ranges, expected_occupied_time)

        expected_schedule = [(0, 1, t1), (1, 2, t4), (2, 4, t2), (4, 5, t1), (5, 6, t4), (6, 7, t1), (7, 11, t3),
                             (11, 12, t1), (12, 13, t1), (13, 14, t4), (15, 16, t1), (16, 17, t4), (17, 19, t2),
                             (19, 20, t1), (20, 21, t4), (21, 22, t1), (24, 25, t1), (25, 26, t4), (27, 28, t1),
                             (28, 32, t3), (32, 33, t1), (33, 34, t4), (34, 35, t1), (35, 36, t4), (36, 37, t1),
                             (37, 39, t2), (39, 40, t1), (40, 41, t4), (42, 43, t1), (45, 46, t1), (46, 47, t4),
                             (47, 49, t2), (49, 50, t1), (50, 51, t4), (51, 52, t1), (52, 56, t3), (56, 57, t1),
                             (57, 58, t1), (58, 59, t4), (60, 61, t1), (61, 62, t4), (62, 64, t2), (64, 65, t1),
                             (65, 66, t4), (66, 67, t1), (69, 70, t1), (70, 71, t4), (72, 73, t1)]

        self.assertEqual(tasks, taskset)
        self.assertTrue(valid)
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)

    def test_invalid_schedule(self):
        t1_subtask = DAGBudgetResourceSubTask(name="T1S", c=2, resources=['1'])
        t1 = PeriodicBudgetResourceDAGTask(name="T1", tasks=[t1_subtask], p=4)
        t2_subtask = DAGBudgetResourceSubTask(name="T2S", c=6, resources=['2'])
        t2 = PeriodicBudgetResourceDAGTask(name="T2", tasks=[t2_subtask], p=12)
        taskset = [t1, t2]

        scheduler = UniResourceCEDFScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(taskset)

        self.assertIsNone(tasks, taskset)
        self.assertFalse(valid)
        self.assertIsNone(schedule)


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


def get_topology_and_tasks():
    line_topology = gen_line_topology()
    slot_size = 0.01
    demands = [('1', '3', 0.6, 5), ('0', '1', 0.8, 10)]
    tasks = []
    for demand in demands:
        protocol = get_protocol_without_rate_constraint(line_topology, demand)
        task = convert_protocol_to_task(demand, protocol, slot_size)
        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, line_topology)
        tasks.append(scheduled_task)

    return line_topology, tasks


class TestMultiResourceNonBlockNPEDF(unittest.TestCase):
    def test_schedule(self):
        test_topology, test_tasks = get_topology_and_tasks()
        scheduler = MultipleResourceNonBlockNPEDFScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(test_tasks, test_topology)

        expected_schedule = [(0, 5, test_tasks[1]), (1, 10, test_tasks[0]), (10, 15, test_tasks[1])]
        self.assertTrue(valid)
        self.assertEqual(len(tasks), len(test_tasks))
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)


class TestMultiResourceBlockNPEDF(unittest.TestCase):
    def test_schedule(self):
        test_topology, test_tasks = get_topology_and_tasks()
        scheduler = MultipleResourceBlockNPEDFScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(test_tasks, test_topology)

        expected_schedule = [(0, 5, test_tasks[1]), (5, 14, test_tasks[0]), (14, 19, test_tasks[1])]
        self.assertTrue(valid)
        self.assertEqual(len(tasks), len(test_tasks))
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)


class TestMultiResourceCEDF(unittest.TestCase):
    def test_schedule(self):
        test_topology, test_tasks = get_topology_and_tasks()
        scheduler = MultipleResourceBlockCEDFScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(test_tasks, test_topology)

        expected_schedule = [(0, 5, test_tasks[1]), (5, 14, test_tasks[0]), (14, 19, test_tasks[1])]
        self.assertTrue(valid)
        self.assertEqual(len(tasks), len(test_tasks))
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)


class TestMultiResourcePBEDF(unittest.TestCase):
    def test_schedule(self):
        test_topology, test_tasks = get_topology_and_tasks()
        scheduler = MultipleResourceConsiderateSegmentPreemptionBudgetScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(test_tasks, test_topology)

        expected_schedule = [(0, 5, test_tasks[1]), (0, 4, test_tasks[0]), (5, 9, test_tasks[0]),
                             (9, 10, test_tasks[0]), (10, 15, test_tasks[1])]
        self.assertTrue(valid)
        self.assertEqual(len(tasks), len(test_tasks))
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)


class TestMultiResourceSegmentPBEDF(unittest.TestCase):
    def test_schedule(self):
        test_topology, test_tasks = get_topology_and_tasks()
        scheduler = MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler()
        [(tasks, schedule, valid)] = scheduler.schedule_tasks(test_tasks, test_topology)

        expected_schedule = [(0, 5, test_tasks[1]), (5, 9, test_tasks[0]), (9, 13, test_tasks[0]),
                             (13, 14, test_tasks[0]), (14, 19, test_tasks[1])]
        self.assertTrue(valid)
        self.assertEqual(len(tasks), len(test_tasks))
        self.assertEqual(len(schedule), len(expected_schedule))

        for expected_entry, computed_entry in zip(expected_schedule, schedule):
            expected_start, expected_end, expected_task = expected_entry
            start, end, task = computed_entry
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(task.name.split("|")[0], expected_task.name)

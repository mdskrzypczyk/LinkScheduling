import unittest
from simulation import gen_topologies, get_protocol
from jobscheduling.protocols import convert_protocol_to_task, schedule_dag_for_resources
from jobscheduling.schedulers.SearchBlockPBEDF import MultipleResourceInconsiderateBlockPreemptionBudgetScheduler,\
    MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler, MultipleResourceConsiderateBlockPreemptionBudgetScheduler,\
    MultipleResourceConsiderateSegmentBlockPreemptionBudgetScheduler, MultipleResourceInconsiderateSegmentPreemptionBudgetScheduler,\
    MultipleResourceConsiderateSegmentPreemptionBudgetScheduler
from jobscheduling.visualize import schedule_and_resource_timelines


class TestMultipleResourceInconsiderateBlockPreemptionBudgetScheduler(unittest.TestCase):
    def test_taskset(self):
        topologies = gen_topologies(4)
        [line_topology, ring_topology, grid_topology, demo_topology] = topologies

        test_demands = [('0', '2', 0.7, 1), ('1', '3', 0.7, 1)]
        protocols = [get_protocol(line_topology, demand) for demand in test_demands]
        tasks = [convert_protocol_to_task(demand, protocol) for demand, protocol in zip(test_demands, protocols)]
        tasks = [schedule_dag_for_resources(task, line_topology)[0] for task in tasks]
        scheduler = MultipleResourceInconsiderateBlockPreemptionBudgetScheduler()
        schedule = scheduler.schedule_tasks(tasks, line_topology)

        task_schedule = schedule[0][1]
        computed_schedule = [(start, end, t.name) for start, end, t in task_schedule]

        expected_schedule = [(0, 1, tasks[0].name + "|0|0"),
                             (1, 2, tasks[0].name + "|0|1"),
                             (2, 3, tasks[1].name + "|0|0"),
                             (3, 4, tasks[1].name + "|0|1")]

        self.assertEqual(computed_schedule, expected_schedule)

    def test_taskset2(self):
        topologies = gen_topologies(4)
        [line_topology, ring_topology, grid_topology, demo_topology] = topologies

        test_demands = [('0', '2', 0.9, 0.1), ('1', '3', 0.9, 0.1)]
        protocols = [get_protocol(line_topology, demand) for demand in test_demands]
        tasks = [convert_protocol_to_task(demand, protocol) for demand, protocol in zip(test_demands, protocols)]
        tasks = [schedule_dag_for_resources(task, line_topology)[0] for task in tasks]
        scheduler = MultipleResourceInconsiderateBlockPreemptionBudgetScheduler()
        schedule = scheduler.schedule_tasks(tasks, line_topology)

        task_schedule = schedule[0][1]
        computed_schedule = [(start, end, t.name) for start, end, t in task_schedule]

        expected_schedule = [(i, i+1, tasks[0].name + "|0|{}".format(i)) for i in range(33)]
        expected_schedule += [(33 + i, 34 + i, tasks[1].name + "|0|{}".format(i)) for i in range(33)]

        self.assertEqual(computed_schedule, expected_schedule)


class TestMultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler(unittest.TestCase):
    def test_taskset(self):
        topologies = gen_topologies(4)
        [line_topology, ring_topology, grid_topology, demo_topology] = topologies

        test_demands = [('0', '2', 0.7, 1), ('1', '3', 0.7, 1)]
        protocols = [get_protocol(line_topology, demand) for demand in test_demands]
        tasks = [convert_protocol_to_task(demand, protocol) for demand, protocol in zip(test_demands, protocols)]
        tasks = [schedule_dag_for_resources(task, line_topology)[0] for task in tasks]
        scheduler = MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler()
        schedule = scheduler.schedule_tasks(tasks, line_topology)

        task_schedule = schedule[0][1]
        computed_schedule = [(start, end, t.name) for start, end, t in task_schedule]

        expected_schedule = [(0, 1, tasks[0].name + "|0|0"),
                             (1, 2, tasks[0].name + "|0|1"),
                             (2, 3, tasks[1].name + "|0|0"),
                             (3, 4, tasks[1].name + "|0|1")]

        self.assertEqual(computed_schedule, expected_schedule)

    def test_taskset2(self):
        topologies = gen_topologies(4)
        [line_topology, ring_topology, grid_topology, demo_topology] = topologies

        test_demands = [('0', '2', 0.9, 0.1), ('1', '3', 0.9, 0.1)]
        protocols = [get_protocol(line_topology, demand) for demand in test_demands]
        tasks = [convert_protocol_to_task(demand, protocol) for demand, protocol in zip(test_demands, protocols)]
        tasks = [schedule_dag_for_resources(task, line_topology)[0] for task in tasks]
        scheduler = MultipleResourceInconsiderateSegmentBlockPreemptionBudgetScheduler()
        schedule = scheduler.schedule_tasks(tasks, line_topology)

        task_schedule = schedule[0][1]
        computed_schedule = [(start, end, t.name) for start, end, t in task_schedule]

        expected_schedule = [(i, i+1, tasks[0].name + "|0|{}".format(i)) for i in range(33)]
        expected_schedule += [(33 + i, 34 + i, tasks[1].name + "|0|{}".format(i)) for i in range(33)]

        self.assertEqual(computed_schedule, expected_schedule)


class TestMultipleResourceInconsiderateSegmentPreemptionBudgetScheduler(unittest.TestCase):
    def test_taskset(self):
        topologies = gen_topologies(4)
        [line_topology, ring_topology, grid_topology, demo_topology] = topologies

        test_demands = [('0', '2', 0.7, 1), ('1', '3', 0.7, 1)]
        protocols = [get_protocol(line_topology, demand) for demand in test_demands]
        tasks = [convert_protocol_to_task(demand, protocol) for demand, protocol in zip(test_demands, protocols)]
        tasks = [schedule_dag_for_resources(task, line_topology)[0] for task in tasks]
        scheduler = MultipleResourceInconsiderateSegmentPreemptionBudgetScheduler()
        schedule = scheduler.schedule_tasks(tasks, line_topology)

        task_schedule = schedule[0][1]
        computed_schedule = [(start, end, t.name) for start, end, t in task_schedule]

        expected_schedule = [(0, 1, tasks[0].name + "|0|0"),
                             (1, 2, tasks[0].name + "|0|1"),
                             (2, 3, tasks[1].name + "|0|0"),
                             (3, 4, tasks[1].name + "|0|1")]

        self.assertEqual(computed_schedule, expected_schedule)

    def test_taskset2(self):
        topologies = gen_topologies(4)
        [line_topology, ring_topology, grid_topology, demo_topology] = topologies

        test_demands = [('0', '2', 0.9, 0.1), ('1', '3', 0.9, 0.1)]
        protocols = [get_protocol(line_topology, demand) for demand in test_demands]
        tasks = [convert_protocol_to_task(demand, protocol) for demand, protocol in zip(test_demands, protocols)]
        tasks = [schedule_dag_for_resources(task, line_topology)[0] for task in tasks]
        scheduler = MultipleResourceInconsiderateSegmentPreemptionBudgetScheduler()
        schedule = scheduler.schedule_tasks(tasks, line_topology)

        task_schedule = schedule[0][1]
        computed_schedule = [(start, end, t.name) for start, end, t in task_schedule]

        expected_schedule = [(i, i+1, tasks[0].name + "|0|{}".format(i)) for i in range(33)]
        expected_schedule += [(33 + i, 34 + i, tasks[1].name + "|0|{}".format(i)) for i in range(33)]

        import pdb
        pdb.set_trace()

        self.assertEqual(computed_schedule, expected_schedule)
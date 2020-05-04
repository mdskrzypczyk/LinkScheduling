import unittest
from jobscheduling.protocolgen import create_protocol
from jobscheduling.protocols import schedule_dag_for_resources, convert_protocol_to_task
from jobscheduling.topology import gen_line_topology
from math import ceil


class TestMapping(unittest.TestCase):
    def test_map_link_protocol(self):
        line_topology = gen_line_topology()
        _, G = line_topology
        test_source = '0'
        test_dest = '1'
        test_fidelity = 0.8
        test_rate = 0.1
        test_slot_size = 0.01

        path = [test_source, test_dest]
        protocol = create_protocol(path, G, test_fidelity, test_rate)
        demand = (test_source, test_dest, test_fidelity, test_rate)
        task = convert_protocol_to_task(demand, protocol, test_slot_size)
        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, line_topology)

        self.assertTrue(correct)

        asap_dec, alap_dec, shift_dec = decoherence_times
        self.assertLessEqual(shift_dec, asap_dec)
        self.assertLessEqual(shift_dec, alap_dec)

        self.assertEqual(len(scheduled_task.subtasks), 1)
        link_task = scheduled_task.subtasks[0]
        self.assertEqual(link_task.name, "L;0;0;1")
        self.assertEqual(link_task.a, 0)
        self.assertEqual(link_task.c, ceil(1 / (protocol.R * test_slot_size)))
        self.assertEqual(link_task.resources, ['0-C0', '0-S0', '1-C0', '1-S0'])

    def test_map_swap_protocol(self):
        line_topology = gen_line_topology()
        _, G = line_topology
        test_source = '0'
        repeater = '1'
        test_dest = '2'
        test_fidelity = 0.7
        test_rate = 0.1
        test_slot_size = 0.01

        path = [test_source, repeater, test_dest]
        protocol = create_protocol(path, G, test_fidelity, test_rate)
        demand = (test_source, test_dest, test_fidelity, test_rate)
        task = convert_protocol_to_task(demand, protocol, test_slot_size)
        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, line_topology)

        self.assertTrue(correct)

        asap_dec, alap_dec, shift_dec = decoherence_times
        self.assertLessEqual(shift_dec, asap_dec)
        self.assertLessEqual(shift_dec, alap_dec)

        expected_subtasks = {"L;0;0;1", "L;1;1;2", "S;0;1"}
        self.assertEqual(len(scheduled_task.subtasks), 3)
        self.assertEqual(set(t.name for t in scheduled_task.subtasks), expected_subtasks)
        self.assertEqual(scheduled_task.c, 11)

        self.assertEqual(set(t.name for t in scheduled_task.sources), {"L;0;0;1", "L;1;1;2"})
        self.assertEqual(set(t.name for t in scheduled_task.sinks), {"S;0;1"})

        sorted_tasks = sorted(scheduled_task.subtasks, key=lambda task: task.a)
        curr_time = 0
        for i, task in enumerate(sorted_tasks):
            self.assertEqual(task.a, curr_time)
            curr_time += task.c

        expected_names = ["L", "L", "S"]
        expected_comps = [5, 5, 1]
        self.assertEqual([t.name[0] for t in sorted_tasks], expected_names)
        self.assertEqual([t.c for t in sorted_tasks], expected_comps)
        expected_resources = [
            ['1-C0', '1-S0', '2-C0', '2-S0'],  # Link Resources
            ['0-C0', '0-S0', '1-C0', '1-S1'],  # Link Resources
            ['1-S0', '1-S1'],  # Swap Resources
        ]
        self.assertEqual([t.resources for t in sorted_tasks], expected_resources)

    def test_map_distillation_protocol(self):
        line_topology = gen_line_topology()
        _, G = line_topology
        test_source = '0'
        repeater = '1'
        test_dest = '2'
        test_fidelity = 0.8
        test_rate = 0.1
        test_slot_size = 0.01

        path = [test_source, repeater, test_dest]
        protocol = create_protocol(path, G, test_fidelity, test_rate)
        demand = (test_source, test_dest, test_fidelity, test_rate)
        task = convert_protocol_to_task(demand, protocol, test_slot_size)
        scheduled_task, decoherence_times, correct = schedule_dag_for_resources(task, line_topology)

        self.assertTrue(correct)

        asap_dec, alap_dec, shift_dec = decoherence_times
        self.assertLessEqual(shift_dec, asap_dec)
        self.assertLessEqual(shift_dec, alap_dec)

        expected_subtasks = {"L;0;0;1", "L;1;1;2", "L;2;0;1", "L;3;1;2", "S;0;1", "S;1;1", "D;0;0;2"}
        self.assertEqual(len(scheduled_task.subtasks), len(expected_subtasks))
        self.assertEqual(set(t.name for t in scheduled_task.subtasks), expected_subtasks)
        self.assertEqual(scheduled_task.c, 33)

        self.assertEqual(set(t.name for t in scheduled_task.sources), {"L;0;0;1", "L;1;1;2", "L;2;0;1", "L;3;1;2"})
        self.assertEqual(set(t.name for t in scheduled_task.sinks), {"D;0;0;2"})

        sorted_tasks = sorted(scheduled_task.subtasks, key=lambda task: (task.a, task.name))
        expected_start_times = [0, 8, 16, 16, 24, 32, 32]
        expected_names = ["L", "L", "L", "S", "L", "D", "S"]
        expected_comps = [8, 8, 8, 1, 8, 1, 1]
        self.assertEqual([t.a for t in sorted_tasks], expected_start_times)
        self.assertEqual([t.name[0] for t in sorted_tasks], expected_names)
        self.assertEqual([t.c for t in sorted_tasks], expected_comps)
        expected_resources = [
            ['1-C0', '1-S0', '2-C0', '2-S0'],   # Link Resources
            ['0-C0', '0-S0', '1-C0', '1-S1'],   # Link Resources
            ['1-C0', '1-S2', '2-C0', '2-S1'],   # Link Resources
            ['1-S0', '1-S1'],                   # Swap Resources
            ['0-C0', '0-S1', '1-C0', '1-S0'],   # Link Resources
            ['0-S0', '0-S1', '2-S0', '2-S1'],   # Distill Resources
            ['1-S0', '1-S2']                    # Swap Resources
        ]
        self.assertEqual([t.resources for t in sorted_tasks], expected_resources)

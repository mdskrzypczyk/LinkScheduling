import unittest
from jobscheduling.qmath import distill_links, undistill_link, undistill_link_even, swap_links, unswap_links, \
    distillations_for_fidelity, fidelity_for_distillations, max_distilled_fidelity
from math import isclose


class TestFunctions(unittest.TestCase):
    def test_distill_and_undistill(self):
        # Base case
        self.assertEqual(distill_links(1, 1), 1)
        self.assertEqual(undistill_link(1, 1), 1)
        self.assertEqual(undistill_link_even(1), 1)

        # More precise case
        test_initial_fidelity = 0.9
        expected_distilled_fidelity = 0.9263959390862944
        self.assertTrue(isclose(distill_links(test_initial_fidelity, test_initial_fidelity),
                                expected_distilled_fidelity))
        self.assertTrue(isclose(undistill_link(expected_distilled_fidelity, test_initial_fidelity),
                                test_initial_fidelity))
        self.assertTrue(isclose(undistill_link_even(expected_distilled_fidelity), test_initial_fidelity))

    def test_swap_and_unswap(self):
        # Base case
        self.assertEqual(swap_links(1, 1), 1)
        self.assertEqual(unswap_links(1), 1)

        # More precise case
        test_initial_fidelity = 0.9949832212875671
        expected_swapped_fidelity = 0.99
        self.assertTrue(isclose(swap_links(test_initial_fidelity, test_initial_fidelity), expected_swapped_fidelity))
        self.assertTrue(isclose(unswap_links(expected_swapped_fidelity), test_initial_fidelity))

    def test_distillations_for_fidelity(self):
        self.assertEqual(distillations_for_fidelity(1, 1), 0)

        Finitial = 0.99
        Ftarget = distill_links(Finitial, Finitial)
        self.assertEqual(distillations_for_fidelity(Finitial, Ftarget), 1)

        Finitial = 0.8
        Ftarget = 1
        self.assertEqual(distillations_for_fidelity(Finitial, Ftarget), float('inf'))

        Ftarget = distill_links(distill_links(Finitial, Finitial), Finitial)
        self.assertEqual(distillations_for_fidelity(Finitial, Ftarget), 2)

    def test_fidelity_for_distillations(self):
        self.assertEqual(fidelity_for_distillations(0, 1), 1)
        Ftarget = distill_links(0.99, 0.99)
        self.assertTrue(isclose(fidelity_for_distillations(1, Ftarget), 0.99, abs_tol=0.00001))

    def test_max_distilled_fidelity(self):
        self.assertEqual(max_distilled_fidelity(1), 1)
        expected_max = 0.9434601571592949
        self.assertEqual(max_distilled_fidelity(0.9), expected_max)

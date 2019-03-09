import unittest
import numpy

from HW.HW2 import support
from HW.HW2 import constants

import logging
log = logging.getLogger("levee.tests.support")

class FlowHeightTest(unittest.TestCase):
	def test_levee_height_lookup(self):
		"""
			Test that a given flow value gives us the correct corresponding required levee height to contain it.
		:return:
		"""
		self.assertAlmostEqual(support.get_required_levee_height(0), 0)
		self.assertAlmostEqual(support.get_required_levee_height(36), 0)
		self.assertAlmostEqual(support.get_required_levee_height(36.9), 0.1)
		self.assertAlmostEqual(support.get_required_levee_height(37), 0.1)
		self.assertAlmostEqual(support.get_required_levee_height(40.9288), 0.1)
		self.assertAlmostEqual(support.get_required_levee_height(40.9290), 0.2)
		self.assertAlmostEqual(support.get_required_levee_height(125), 2.1)
		self.assertAlmostEqual(support.get_required_levee_height(835), 14.9)
		self.assertAlmostEqual(support.get_required_levee_height(841), 15.0)
		self.assertAlmostEqual(support.get_required_levee_height(842), 16.0)  # when it's out of spec, it's max height plus one

	def test_levee_overtopping(self):

		self.assertFalse(support.levee_is_overtopped(0, 0))  # a flow of 0 is below the toe of the levee
		self.assertFalse(support.levee_is_overtopped(5, 0))  # a flow of 5 is below the toe of the levee
		self.assertFalse(support.levee_is_overtopped(36, 0))

		self.assertTrue(support.levee_is_overtopped(37, 0))
		self.assertFalse(support.levee_is_overtopped(37, 0.1))

		self.assertTrue(9999, 0.1)
		self.assertTrue(9999, 15)

	def test_vectorized_overtopping(self):
		probabilities = numpy.array([0.25, 0.74, 0.01])
		flows = numpy.array([1, 50, 99])
		total_cost = support.get_overtopping_costs(flows, 0, probabilities)

		print("Total Cost: {}".format(total_cost))
		self.assertAlmostEqual(total_cost, 7500000.0)


class ScenarioTest(unittest.TestCase):
	def setUp(self):
		self.scenarios = support.get_scenarios(number_of_stages=constants.NUMBER_TIME_STEPS)

	def test_probabilities_equal_one(self):
		"""
			checks that after we run the bayesian probabilities, they total 1 for each stage.
			If they don't, then we have a big problem since that means we're going to scale
			things past the actual value. Failing as of 3/9/2019, meaning we need to rework
			our Bayesian Updating code...
		:return:
		"""
		probabilities = []
		for scenario in self.scenarios:
			probabilities.append(scenario.bayesian_probabilities)

		for stage in range(constants.NUMBER_TIME_STEPS):
			stage_probabilities = numpy.array(probabilities[:][stage])
			log.debug("Stage {} probabilities: {}".format(stage, stage_probabilities))
			self.assertAlmostEqual(1, stage_probabilities.sum())





import unittest

from HW.HW2 import dp


class DPTest(unittest.TestCase):
	def setUp(self):

		decision_variable = dp.DecisionVariable("Time on Course", options=[1, 2, 3, 4])
		state_variable = dp.StateVariable("Days Spent Studying", values=[1, 2, 3, 4, 5, 6, 7])

		benefit_list = [
			[3, 5, 6, 7],
			[5, 6, 7, 9],
			[2, 4, 7, 8],
			[6, 7, 9, 9],
		]

		dynamic_program = dp.DynamicProgram(timestep_size=1, time_horizon=4)
		dynamic_program.build_stages(name_prefix="Course")
		for index, stage in enumerate(dynamic_program.stages):
			stage.cost_benefit_list = benefit_list[index]  # set the benefit options for each stage

		dynamic_program.decision_variable = decision_variable
		dynamic_program.add_state_variable(state_variable)

		dynamic_program.run()

	def test_basic(self):
		self.assertTrue(True)  # just want it to run setUp right now
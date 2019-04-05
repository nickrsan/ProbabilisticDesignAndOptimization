import pytest

from .. import support

import dypy

def test_single_var_probabilistic():
	"""
		Meant to be for the version that has a single state variable, but the objective
		function handles variation
	:return:
	"""
	height_var = dypy.StateVariable("height")
	decision_var = dypy.DecisionVariable(min=0.0, max=10.0)
	decision_var.related_state = height_var  # tell the decision variable which state it impacts

	dynamic_program = dypy.DynamicProgram(objective_function=support.objective_function,
										state_variables=(height_var,),
										decision_variable=decision_var)

	dynamic_program.run()  # runs the backward recursion

	# runs the forward method to obtain choices
	optimal_path = dynamic_program.get_optimal_values()

	assert False is True  # for now just make the test fail until we figure out what the value should be


def test_multi_state_probabilistic():
	"""
		Meant to be for the version that has a all three state variables
	:return:
	"""
	height_var = dypy.StateVariable("height")
	flow_var = dypy.StateVariable("peak_flow")
	variance_var = dypy.StateVariable("variance")
	decision_var = dypy.DecisionVariable(min=0.0, max=10.0)
	decision_var.related_state = height_var  # tell the decision variable which state it impacts

	dynamic_program = dypy.DynamicProgram(objective_function=support.objective_function,
										state_variables=(height_var, flow_var, variance_var),
										decision_variable=decision_var)

	dynamic_program.run()  # runs the backward recursion

	# runs the forward method to obtain choices
	optimal_path = dynamic_program.get_optimal_values()

	assert False is True  # for now just make the test fail until we figure out what the value should be
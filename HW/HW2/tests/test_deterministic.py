import logging_test
import logging
log = logging.getLogger("hw2.deterministic")


import pytest
import numpy

from .. import support
from .. import constants

import dypy


def test_deterministic():
	"""
		Run it with one variable - uses a slightly probabilistic method, but only a
	:return:
	"""

	num_state_values = int((constants.MAXIMUM_LEVEE_HEIGHT - constants.INITIAL_LEVEE_HEIGHT) / constants.LEVEE_HEIGHT_INCREMENT) + 1
	state_values = numpy.linspace(constants.INITIAL_LEVEE_HEIGHT, constants.MAXIMUM_LEVEE_HEIGHT, num_state_values, endpoint=True)
	height_var = dypy.StateVariable("height", values=state_values, initial_state=constants.INITIAL_LEVEE_HEIGHT)
	decision_var = dypy.DecisionVariable(name="build_increment", minimum=constants.INITIAL_LEVEE_HEIGHT, maximum=constants.MAXIMUM_UPGRADE_LEVEE_HEIGHT, step_size=constants.LEVEE_HEIGHT_INCREMENT)
	decision_var.related_state = height_var  # tell the decision variable which state it impacts

	dynamic_program = dypy.DynamicProgram(objective_function=support.objective_function,
										state_variables=(height_var,),
										decision_variable=decision_var,
										calculation_function=dypy.MINIMIZE,
										prior=dypy.DiscountedSimplePrior,
										timestep_size=constants.TIME_STEP_SIZE,
										time_horizon=constants.TIME_HORIZON,
										discount_rate=constants.DISCOUNT_RATE
	)
	dynamic_program.build_stages(name_prefix="Year")
	dynamic_program.run()  # runs the backward recursion

	for stage in dynamic_program.stages:
		log.info("Stage {} choice: {}. Cost: {}".format(stage.number, stage.decision_amount, stage.future_value_of_decision))

	assert False is True  # for now just make the test fail until we figure out what the value should be


def test_multivariable():
	"""
		Use a multi-state variable formulation and get the path for each climate scenario
	:return:
	"""

	scenario_names = ('A', 'B', 'C', 'D', 'E', 'F')

	num_state_values = int((constants.MAXIMUM_LEVEE_HEIGHT - constants.INITIAL_LEVEE_HEIGHT) / constants.LEVEE_HEIGHT_INCREMENT) + 1
	state_values = numpy.linspace(constants.INITIAL_LEVEE_HEIGHT, constants.MAXIMUM_LEVEE_HEIGHT, num_state_values, endpoint=True)
	height_var = dypy.StateVariable("initial_height", values=state_values, initial_state=constants.INITIAL_LEVEE_HEIGHT, availability_function=numpy.isclose)
	scenario_var = dypy.StateVariable("scenario_name", values=scenario_names)
	decision_var = dypy.DecisionVariable(name="incremental_height", minimum=constants.INITIAL_LEVEE_HEIGHT, maximum=constants.MAXIMUM_UPGRADE_LEVEE_HEIGHT, step_size=constants.LEVEE_HEIGHT_INCREMENT)
	decision_var.related_state = height_var  # tell the decision variable which state it impacts

	dynamic_program = dypy.DynamicProgram(objective_function=support.total_costs_of_choice_multivariable,
										state_variables=(height_var, scenario_var),
										decision_variable=decision_var,
										calculation_function=dypy.MINIMIZE,
										prior=dypy.DiscountedSimplePrior,
										timestep_size=constants.TIME_STEP_SIZE,
										time_horizon=constants.TIME_HORIZON,
										discount_rate=constants.DISCOUNT_RATE
	)
	dynamic_program.build_stages(name_prefix="Year")
	dynamic_program.run()  # runs the backward recursion

	for scenario in scenario_names:
		log.info("\nScenario {}".format(scenario))
		dynamic_program.state_variables[1].initial_state = scenario
		dynamic_program.state_variables[1].reset_state()
		dynamic_program.state_variables[0].reset_state()  # force the globally optimum first stage choice with uncertain scenario outcome
		dynamic_program.state_variables[0].current_state = dynamic_program.stages[0].decision_amount  # force the globally optimum first stage choice with uncertain scenario outcome
		dynamic_program.stages[1].get_optimal_values()  # then start the

		for stage in dynamic_program.stages:
			log.info("Stage {} choice: {}. Cost: {}".format(stage.number, stage.decision_amount,
															stage.future_value_of_decision))


	assert False is True  # for now just make the test fail until we figure out what the value should be


def test_multivariable_no_first_stage_lock():
	"""
		Use a multi-state variable formulation and get the path for each climate scenario
	:return:
	"""

	scenario_names = ('A', 'B', 'C', 'D', 'E', 'F')

	num_state_values = int((constants.MAXIMUM_LEVEE_HEIGHT - constants.INITIAL_LEVEE_HEIGHT) / constants.LEVEE_HEIGHT_INCREMENT) + 1
	state_values = numpy.linspace(constants.INITIAL_LEVEE_HEIGHT, constants.MAXIMUM_LEVEE_HEIGHT, num_state_values, endpoint=True)
	height_var = dypy.StateVariable("initial_height", values=state_values, initial_state=constants.INITIAL_LEVEE_HEIGHT)
	scenario_var = dypy.StateVariable("scenario_name", values=scenario_names)
	decision_var = dypy.DecisionVariable(name="incremental_height", minimum=constants.INITIAL_LEVEE_HEIGHT, maximum=constants.MAXIMUM_UPGRADE_LEVEE_HEIGHT, step_size=constants.LEVEE_HEIGHT_INCREMENT)
	decision_var.related_state = height_var  # tell the decision variable which state it impacts

	dynamic_program = dypy.DynamicProgram(objective_function=support.total_costs_of_choice_multivariable,
										state_variables=(height_var, scenario_var),
										decision_variable=decision_var,
										calculation_function=dypy.MINIMIZE,
										prior=dypy.DiscountedSimplePrior,
										timestep_size=constants.TIME_STEP_SIZE,
										time_horizon=constants.TIME_HORIZON,
										discount_rate=constants.DISCOUNT_RATE
	)
	dynamic_program.build_stages(name_prefix="Year")
	dynamic_program.run()  # runs the backward recursion

	for scenario in scenario_names:
		log.info("\nScenario {}".format(scenario))
		dynamic_program.state_variables[1].initial_state = scenario
		dynamic_program.get_optimal_values()  # then start the

		for stage in dynamic_program.stages:
			log.info("Stage {} choice: {}. Cost: {}".format(stage.number, stage.decision_amount,
															stage.future_value_of_decision))


	assert False is True  # for now just make the test fail until we figure out what the value should be
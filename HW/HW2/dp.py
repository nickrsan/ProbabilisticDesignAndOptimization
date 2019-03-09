"""
	Dynamic Program class - implements only backward dynamic programming.
	Ideal usage is to have an objective function defined by the user, however they'd like.

	The user defines as many StateVariable instances as they have state variables in their DP and they define
	a DecisionVariable. The objective function should be prepared to take arguments with keyword values for each of
	these, where the keyword is determined by the name attribute on each instance. It then returns a value.

	For situations with multiple state variables, we have reducers, which, prior to minimization or maximization
	will reduce the number of state variables to one so that we can get a single F* for each input scenario.
	This is a class Reducer with a defined interface, and can then be extended. For our levee problem, we have a
	ProbabilisticReducer which keeps track of the probability of each set of state variables and can collapse by
	a master variable. Probabilities must be provided by the user.

	At that point, usage for the levee problem should be something like:

	```
	import support  # user's extra code to support the objective function
	import dp

	objective_function = support.objective_function
	height_var = dp.StateVariable("height")
	flow_var = dp.StateVariable("peak_flow")
	variance_var = dp.StateVariable("variance")
	decision_var = dp.DecisionVariable()
	decision_var.related_state = height_var  # tell the decision variable which state it impacts

	# TODO: Make sure to check how we can make the decision variable properly interact with the state variable - thinking
			# of making sure that the decision variable adds to the correct state variable items
	# TODO: Missing plan for how we assign probabilities here - needs to be incorporated somewhere
	# TODO: Missing choice constraints (eg, min selection at stage ___ is ___)

	dynamic_program = dp.DynamicProgram(objective_function=objective_function,
										state_variables=(height_var, flow_var, variance_var),
										decison_variable=decision_var)
	dynamic_program.optimize()  # runs the backward recursion
	dynamic_program.get_optimal_values()  # runs the forward method to obtain choices
	```
"""

import numpy
import logging

import support

log = logging.getLogger("dp")

MAXIMIZE = max
MINIMIZE = min


class Reducer(object):
	"""
		Reduces multiple state variables to a single state variable so we can just minimize
	"""
	pass


class VariableReducer(Reducer):
	"""
		Given a StateVariable, reduces the table size by collapsing all other variables - can do this by min/max/mean/sum
		of all options.

		Saving implementation here until after we have a better sense for how the rest of this will be implemented
	"""
	def __init__(self, variable, stage):
		self.variable = variable  # reference to StateVariable object
		self.stage = stage  # reference to Stage object


class ProbabilisticReducer(Reducer):
	"""
		Given a StateVariable to process (S), and a set of StateVariables to hold constant (Cs), reduces S for each
		combination of Cs by multiplying the objective values in the records for S by their probabilities and summing them.

		We should be able to actually just make this have a single column for probabilities so that we can do the same
		thing we planned to do for the variable reducer and just (ignoring the first paragraph of this docstring)
		select a master variable, get all rows for it, multiply those rows by the probability field, and sum them up.
	"""
	pass


class StateVariable(object):
	"""
		Not sure what I'm going to do with this yet, but I think we'll need it in order to have rows for interactions
		between multiple state variables.

		When we go to use all the state variables together, we'll need to discretize them each, and then we'll need to combine
		them to get all the rows in the table for each stage. Assuming we have some attribute .discretized that contains
		all the values for this state variable and that the DynamicProgram class has a list of these state variables called
		.state_variables, we can get all possible combinations for generating a row using `itertools.product(*[var.discretized for var in self.state_variables])`
		Note the asterisk at the front, which takes that list and expands it so each one is an individual argument to
		itertools.product

		We can then use this by taking the name attribute of the state variable and passing it as the kwarg to the objective
		function along with the discretized value. So then we have a DP class that accepts a list of variables, an objective
		function, and then a preprocessor function that handles aggregation of choices between filling in the matrix
		and use of minimization (a function that accepts the array and reduces the choices so that we can actually minimize
		or maximize. We need this to reduce multiple state variables to a single state variable.
	"""

	def __init__(self, name, values):
		self.name = name
		self.values = values

		self.column_index = None  # this will be set by the calling DP - it indicates what column in the table has this information


class DecisionVariable(object):
	def __init__(self, name, related_state=None, minimum=None, maximum=None, step_size=None, options=None):
		"""
			We'll use this to manage the decision variable - we'll need columns for each potential value here
		:param name:
		:param state: the StateVariable object that this DecisionVariable directly feeds back on
		"""
		self.name = name
		self.related_state = related_state

		self._min = minimum
		self._max = maximum
		self._step_size = step_size
		self._options = options
		if options:
			self._user_set_options = True  # keep track so we can zero it out later if they set min/max/stepsize params
		else:
			self._user_set_options = False

		self.constraints = {}

	# we have all of these simple things as @property methods instead of simple attributes so we can
	# make sure to have the correct behaviors if users set the options themselves
	@property
	def minimum(self):
		return self._min

	@property
	def maximum(self):
		return self._max

	@property
	def step_size(self):
		return self._step_size

	@minimum.setter
	def minimum(self, value):
		self._min = value
		self._reset_options()

	@maximum.setter
	def maximum(self, value):
		self._max = value
		self._reset_options()

	@step_size.setter
	def step_size(self, value):
		self._step_size = value
		self._reset_options()

	def _reset_options(self):
		if not self._user_set_options:
			self._options = None  # if we change any of the params, clear the options

	@property
	def options(self):
		if self._options:  # if they gave us options
			return self._options
		elif self._min and self._max and self.step_size:
			if type(self._min) == "int" and type(self._max) == "int" and type(self.step_size) == "int":  # if they're all integers we'll use range
				self._options = range(self._min, self._max, self.step_size)  # cache it so next time we don't have to calculate
			else:
				# the `num` param here just transforms step_size to its equivalent number of steps for linspace. Add 1 to capture accurate spacing with both start and endpoints
				self._options = numpy.linspace(start=self._min, stop=self._max, num=int((self._max-self._min)/self.step_size)+1, endpoint=True)

			self._user_set_options = False
			return self._options

		raise ValueError("Can't get DecisionVariable options - need either explicit options (.options) or a minimum value, a maximum value, and a step size")

	@options.setter
	def options(self, value):
		self._options = value
		self._user_set_options = True

	def add_constraint(self, stage, value):
		"""
			Want to figure out a way here to store also whether this constraint is a minimum or a maximum value constraint.
			Need to think how we'd handle that behavior
		:param stage:
		:param value:
		:return:
		"""
		pass


class DynamicProgram(object):
	"""
		This object actually runs the DP - doesn't currently support multiple decision variables or probabilities.

		Currently designed to only handle backward DPs
	"""

	def __init__(self, objective_function, timestep_size, time_horizon, discount_rate, state_variables=None, selection_constraints=None, decision_variables=None):
		"""

		:param objective_function: What function are we using to evaluate? Basically, is this a maximization (benefit)
		 or minimization (costs) setup. Provide the function object for max or min. Provide the actual `min` or `max functions
		 (don't run it, just the name) or if convenient, use the shortcuts dp.MINIMIZE or dp.MAXIMIZE

		:param selection_constraints: Is there a minimum value that must be achieved in the selection?
				If so, this should be a list with the required quantity at each time step

		:param decision_variables: list of DecisionVariable objects

		:param discount_rate: give the discount rate in "annual" units. Though timesteps don't need to be in years, think
			of the discount rate as applying per smallest possible timestep size, so if your timestep_size is 40, then
			your discount rate will be transformed to cover 40 timesteps (compounding).
		"""
		self.stages = []
		self.timestep_size = timestep_size
		self.time_horizon = time_horizon
		self.discount_rate = discount_rate

		if not state_variables:
			self.state_variables = []
		else:
			self.state_variables = list(state_variables)  # coerce to list in case they gave us something immutable like a tuple
			self._index_state_variables()

		# set up decision variables passed in
		if not decision_variables:
			self.decision_variable = None
		else:
			self.decision_variable = decision_variable

		# Calculation Function
		self.objective_function = objective_function

		if self.objective_function not in (max, min, MAXIMIZE, MINIMIZE):
			raise ValueError("Calculation function must be either 'max' or 'min' or one of the aliases in this package of dp.MAXIMIZE or dp.MINIMIZE")

		# make values that we use as bounds in our calculations - when maximizing, use a negative number, and when minimizing, get close to infinity
		# we use this for any routes through the DP that get excluded
		if self.objective_function is max:
			self.exclusion_value = -1  # just need it to be less
		elif self.objective_function is min:
			self.exclusion_value = 9223372036854775808  # max value for a signed 64 bit int - this should force it to not be selected in minimization

	def add_state_variable(self, variable):
		"""

		:param variable: A StateVariable object - afterward, will be available in .state_variables
		:return:
		"""
		if not isinstance(variable, StateVariable):
			raise ValueError("Provided variable must be a StateVariable object. Can't add variable of type {} to DP".format(type(variable)))

		self.state_variables.append(variable)
		self._index_state_variables()  # make sure to reindex the variables when we add one

	def _index_state_variables(self):
		for index, variable in enumerate(self.state_variables):
			if not isinstance(variable, StateVariable):  # this is a bit silly to have this check twice, but this method checks it even if the user passes a list of StateVariables
				raise ValueError("Provided variable must be a StateVariable object. Can't add variable of type {} to DP".format(type(variable)))

			variable.column_index = index  # tell the variable what column it is

	def add_stage(self, name):
		stage = Stage(name=name)

		self.stages.append(stage)
		self._index_stages()

	def _index_stages(self):

		# assigning .number allows us to have constraints on the number of items selected at each stage
		if len(self.stages) > 1:  # if there are at least two, then set the next and previous objects on the first and last
			self.stages[0].next = self.stages[1]
			self.stages[0].number = 0
			self.stages[-1].previous = self.stages[-2]
			self.stages[-1].number = len(self.stages) - 1

		for i, stage in enumerate(self.stages[1:-1]):  # for all stages except the first and last, then we set both next and previous
			self.stages[i].next = self.stages[i+1]
			self.stages[i].previous = self.stages[i-1]
			self.stages[i].number = i

	def build_stages(self, name_prefix="Step"):
		"""
			Make a stage for every timestep
		:param name_prefix: The string that will go before the stage number when printing information
		:return:
		"""
		for stage_id in range(0, self.time_horizon+1, self.timestep_size):
			self.add_stage(name="{} {}".format(name_prefix, stage_id))

	def run(self):

		if not self.decision_variable or len(self.state_variables) == 0:
			raise ValueError("Decision Variable and State Variables must be attached to DynamicProgram before running. Use .add_state_variable to attach additional state variables, or set .decision_variable to a DecisionVariable object first")

		# build a matrix where everything is 0  - need to figure out what the size of the x axis is
		# this matrix should just have a column for each timestep (we'll pull these out later), which will then be used by
		# each stage to actually build its own matrix
		rows = int(self.time_horizon/self.timestep_size)  # this is the wrong way to do this - the matrix should
		matrix = numpy.zeros((rows, ))

		for stage in range(rows):
			for index, row in enumerate(matrix):
				matrix[index][stage] = support.present_value(index, year=stage*self.timestep_size, discount_rate=self.discount_rate )

		# This next section is old code from a prior simple DP - it will be removed, but was how the set of stages was
		# built previously so I can see what the usage was like while building this for multiple objectives
		stages = []
		for year in range(rows):
			cost_list = matrix_array[1:, year]  # pull the column out of the matrix corresponding to this year - remove the 0 value first row (should look into how this is getting there)
			year_stage = Stage(name="Year {}".format(year), cost_benefit_list=list(cost_list), calculation_function=min, selection_constraints=required)
			year_stage.max_selections = needed_trucks
			year_stage.number = year
			stages.append(year_stage)

		# initiate the optimization and retrieval of the best values
		self.stages[-1].optimize()
		self.stages[0].get_optimal_values()


class Stage(object):
	def __init__(self, name, cost_benefit_list, parent_dp, max_selections=7, previous=None, next_stage=None):
		"""

		:param name:
		:param cost_benefit_list: an iterable containing benefit or cost values at each choice step
		:param max_selections: How many total items are we selecting?
		:param previous: The previous stage, if one exists
		:param next_stage: The next stage, if one exists
		"""
		self.name = name
		self.parent_dp = parent_dp
		self.cost_benefit_list = cost_benefit_list
		self.max_selections = max_selections
		self.next = next_stage
		self.previous = previous
		self.matrix = None  # this will be created from the parameters when .optimize is run
		self.number = None

		self.pass_data = []
		self.choices_index = []

	def optimize(self, prior=None):

		if self.parent_dp.selection_constraints and self.number is None:
			raise ValueError("Stage number(.number) must be identified in sequence in order to use selection constraints")

		# first, we need to get the data from the prior stage - if nothing was passed in, we're on the first step and can skip some things
		if prior is None:
			self.pass_data = self.cost_benefit_list
			if self.parent_dp.selection_constraints:  # then we have selections constraints
				for row_index, row_value in enumerate(self.pass_data):
					if row_index >= len(self.pass_data) - self.parent_dp.selection_constraints[self.number]:
						self.pass_data[row_index] = self.parent_dp.exclusion_value
		else:
			# we're on a stage after the first stage
			# make an empty matrix
			self.matrix = [[0 for i in range(len(self.cost_benefit_list) + 1)] for i in range(self.max_selections)] # adding 1 because we need a 0th column

			# add the priors to the first column
			for index, value in enumerate(prior):
				self.matrix[index][0] = value  # set the values in the first column

			# add the benefit values - these go on the -1:1 line basically
			for index, value in enumerate(self.cost_benefit_list):
				self.matrix[index][index+1] = value

			# set up maximum selection constraints based on what will have been required previously
			if self.parent_dp.selection_constraints:  # then we have selections constraints
				for row_index, row_value in enumerate(self.matrix):
						if row_index >= len(self.matrix) - self.parent_dp.selection_constraints[self.number]:  # if the row is less than the constrained amount for this stage
							for column_index, column_value in enumerate(self.matrix[row_index]):
								self.matrix[row_index][column_index] = self.parent_dp.exclusion_value

			# now calculate the remaining values
			for row_index, row_value in enumerate(self.matrix):
				for column_index, column_value in enumerate(self.matrix[row_index]):
					if column_value == 0 and row_index - column_index >= 0:  # we only need to calculate fields that meet this condition - the way it's calculated, others will cause an IndexError anyway
						stage_value = self.matrix[column_index-1][column_index]  # the value for this stage is on 1:1 line, so, it is where the indices are equal
						prior_value = self.matrix[row_index-column_index][0]
						if stage_value != 0 and prior_value != 0:  # if both are defined and not None
							self.matrix[row_index][column_index] = stage_value + prior_value

			# now remove the Nones so we can call min/max in Python 3
			for row_index, row_value in enumerate(self.matrix):
				for column_index, column_value in enumerate(self.matrix[row_index]):
					if column_value == 0:
						self.matrix[row_index][column_index] = self.parent_dp.exclusion_value  # setting to exclusion value makes it unselected still

			self.pass_data = [self.parent_dp.calculation_function(row) for row in self.matrix]  # sum the rows and find the max
			self.choices_index = [row.index(self.parent_dp.calculation_function(row)) for row in self.matrix]  # get the column of the min/max value

		if self.previous:
			self.previous.optimize(self.pass_data)  # now run the prior stage

	def get_optimal_values(self, prior=0):
		"""
			After running the backward DP, moves forward and finds the best choices at each stage.
		:param prior: The value of the choice at each stage
		:return: 
		"""

		amount_remaining = self.max_selections - prior
		if amount_remaining > 0:
			available_options = self.pass_data[:amount_remaining]  # strip off the end of it to remove values that we can't use
			best_option = available_options[-1]  # get the last value
			row_of_best = available_options.index(best_option)  # now we need the actual row to use in the matrix

			if self.matrix:
				column_of_best = self.matrix[row_of_best].index(best_option)  # get the column of the best option - also the number of days
			else:  # this triggers for the last item, which doesn't have a matrix, but just a costs list
				column_of_best = row_of_best + 1
		else:
			best_option = 0
			column_of_best = 0

		#if self.selection_constraints:
		#	number_of_items = max([0, column_of_best - self.selection_constraints[self.number]])  # take the max with 0 - if it's negative, it should be 0
		#else:
		number_of_items = column_of_best
		print("{} - Number of items: {}, Total Cost/Benefit: {}".format(self.name, number_of_items, best_option))

		if self.next:
			self.next.get_optimal_values(prior=number_of_items+prior)



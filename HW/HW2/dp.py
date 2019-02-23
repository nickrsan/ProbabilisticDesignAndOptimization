

MAXIMIZE = max
MINIMIZE = min


class DynamicProgram(object):
	"""
		This object actually runs the DP - doesn't currently support multiple decision variables or probabilities.

		Currently, this code uses a list-based method for each stage of the DP, but it could likely be sped up significantly
		by converting this to use numpy instead. This will be a future upgrade.
	"""

	def __init__(self, calculation_function):
		"""

		:param calculation_function: What function are we using to evaluate? Basically, is this a maximization (benefit)
		 or minimization (costs) setup. Provide the function object for max or min. Provide the actual `min` or `max functions
		 (don't run it, just the name) or if convenient, use the shortcuts dp.MINIMIZE or dp.MAXIMIZE
		"""
		self.stages = []

		self.calculation_function = calculation_function

		if self.calculation_function not in (max, min, MAXIMIZE, MINIMIZE):
			raise ValueError("Calculation function must be either 'max' or 'min' or one of the aliases in this package of dp.MAXIMIZE or dp.MINIMIZE")

		if self.calculation_function is max:
			self.exclusion_value = -1  # just need it to be less
		elif self.calculation_function is min:
			self.exclusion_value = 9223372036854775808  # max value for a signed 64 bit int - this should force it to not be selected in minimization

	def add_stage(self):
		stage = Stage()

		self.stages.append(stage)
		self._index_stages()

	def _index_stages(self):
		if len(self.stages) > 1: # if there are at least two, then set the next and previous objects on the first and last
			self.stages[0].next = self.stages[1]
			self.stages[-1].previous = self.stages[-2]

		for i, stage in enumerate(self.stages[1:-1]):  # for all stages except the first and last, then we set both next and previous
			self.stages[i].next = self.stages[i+1]
			self.stages[i].previous = self.stages[i-1]


class Stage(object):
	def __init__(self, name, cost_benefit_list, parent_dp, max_selections=7, previous=None, next=None, selection_constraints=None):
		"""

		:param name:
		:param cost_benefit_list: an iterable containing benefit or cost values at each choice step
		:param max_selections: How many total items are we selecting?
		:param previous: The previous stage, if one exists
		:param next: The next stage, if one exists
		:param selection_constraints: Is there a minimum value that must be achieved in the selection? If so, this should be a list with the required quantity at each time step
		"""
		self.name = name
		self.parent_dp = parent_dp
		self.cost_benefit_list = cost_benefit_list
		self.max_selections = max_selections
		self.next = next
		self.previous = previous
		self.matrix = None  # this will be created from the parameters when .optimize is run
		self.selection_constraints = selection_constraints
		self.number = None

	def optimize(self, prior=None):

		if self.selection_constraints and self.number is None:
			raise ValueError("Stage number(.number) must be identified in sequence in order to use selection constraints")

		# first, we need to get the data from the prior stage - if nothing was passed in, we're on the first step and can skip some things
		if prior is None:
			self.pass_data = self.cost_benefit_list
			if self.selection_constraints:  # then we have selections constraints
				for row_index, row_value in enumerate(self.pass_data):
					if row_index >= len(self.pass_data) - self.selection_constraints[self.number]:
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
			if self.selection_constraints:  # then we have selections constraints
				for row_index, row_value in enumerate(self.matrix):
						if row_index >= len(self.matrix) - self.selection_constraints[self.number]:  # if the row is less than the constrained amount for this stage
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

	def get_optimal_values(self, prior=None):
		if not prior:
			prior = 0

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
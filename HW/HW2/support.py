import math
import bisect  # we'll use this to find appropriate z_score probabilities
import logging
import sys

import numpy
from scipy import stats

from . import constants

# set up logging
root_log = logging.getLogger()
if len(root_log.handlers) == 0:  # if we haven't already set up logging, set it up
	root_log.setLevel(logging.DEBUG)
	log_stream_handler = logging.StreamHandler(sys.stdout)
	log_stream_handler.setLevel(logging.DEBUG)
	log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	log_stream_handler.setFormatter(log_formatter)
	root_log.addHandler(log_stream_handler)

log = logging.getLogger("levee.support")

class Scenario(object):
	def __init__(self, name,
				 initial_probability,
				 mean_peak_growth,
				 sd_peak_growth,
				 sd_sd_growth=constants.SIGMA_OF_SIGMA,
				 initial_mean=constants.INITIAL_MEAN_OF_ANNUAL_FLOOD_FLOW,
				 initial_sd=constants.INITIAL_SD_OF_ANNUAL_FLOOD_FLOW,
				 number_of_stages=constants.NUMBER_TIME_STEPS):

		self.name = name
		self.log = logging.getLogger("levee.support.scenario_{}".format(name))
		self.initial_probability = initial_probability
		self.mean_peak_growth = mean_peak_growth  # mean peak flood growth by decade
		self.sd_peak_growth = sd_peak_growth  # standard deviation of peak flood growth by decade
		self.sd_sd_growth = sd_sd_growth  # growth of the standard deviation itself
		self.number_of_stages = number_of_stages+1  # add 1 because stage 0 doesn't count here

		# these are the annual flow paremeters in the initial period, but *NOT* the
		# parameters for the PEAK flows. We use these to construct a lognormal
		# distribution, to then get the peak flows out of it. The flows are lognormal
		# but the probability distributions (for the z scores when updating??) are
		# normal. This is something that needs confirming. See page 6 of Rui's paper
		self.initial_mean = initial_mean
		self.initial_sd = initial_sd
		self.initial_probability = initial_probability

		# following variables are all keyed by stage number
		self.mean_at_stage = [initial_mean,] * self.number_of_stages
		self.sd_at_stage = [initial_sd,] * self.number_of_stages
		self.mean_z_scores = [0,] * self.number_of_stages
		self.sd_z_scores = [0,] * self.number_of_stages
		self.mean_probabilities = [0,] * self.number_of_stages
		self.sd_probabilities = [0,] * self.number_of_stages
		self.bayesian_numerators = [0,] * self.number_of_stages
		self.bayesian_probabilities = [self.initial_probability, ] * self.number_of_stages
		#

		self.probability_distribution_discretization = numpy.linspace(start=constants.PROBABILITY_DISTRIBUTION_LIMITS[0],
																	  stop=constants.PROBABILITY_DISTRIBUTION_LIMITS[1],
																	  num=constants.PROBABILITY_DISTRIBUTION_DISCRETIZATION_UNITS)
		self.probabilities = {}  # we'll index and store total probabilities here for each item in the probability discretization

		self.log.info("Indexing probabilities")
		self._index_z_probabilities()
		for stage in range(1, self.number_of_stages):  # fill the stage data
			self.new_stage_observation(stage)

	def _index_z_probabilities(self):
		total_items = len(self.probability_distribution_discretization)
		for i, lower_z in enumerate(self.probability_distribution_discretization):
			if i+1 == total_items:  # don't process the last item - it's the top of a range
				break

			upper_z = self.probability_distribution_discretization[i+1]
			lower_z_cum_prob = stats.norm.cdf(lower_z)
			upper_z_cum_prob = stats.norm.cdf(upper_z)
			self.probabilities[lower_z] = upper_z_cum_prob - lower_z_cum_prob

	def get_probability(self, prob_z_score):

		# make sure it's in range - if it's greater than the max, use the max. If it's less than the min, use the min
		prob_z_score = min(prob_z_score, constants.PROBABILITY_DISTRIBUTION_LIMITS[1])
		prob_z_score = max(prob_z_score, constants.PROBABILITY_DISTRIBUTION_LIMITS[0])

		# we subset the discretization because the last item doesn't actually have a key - it's rolled into the previous one
		probability_index = bisect.bisect_right(self.probability_distribution_discretization[:-1], prob_z_score) - 1 # get the location of the min z_score involved
		min_z_score = self.probability_distribution_discretization[probability_index]
		return self.probabilities[min_z_score]  # returns the probability for that range

	def _value_at_decade(self, value, decade, growth):
		"""
			Provides a compounded, continuous decadal growth.
		:param value:  The initial value at time 0
		:param decade:  How many decades in the future we're calculating for
		:param growth:  The growth rate per decade
		:return:  Future value with decadally compounded growth
		"""
		return value*math.exp(growth*decade)

	def mean_at_decade(self, decade):
		"""
			I thought maybe I wasn't calculating this correctly - but maybe
			that was the old version - not sure. Worth verifying that I'm calculating
			the correct starting values for each scenario
		:param decade:
		:return:
		"""
		return self._value_at_decade(self.initial_mean, decade, self.mean_peak_growth)

	def sd_at_decade(self, decade):
		"""
			TODO: Talk to someone about how we grow the SD given this and the SIGMA_OF_SIGMA
		:param decade:
		:return:
		"""
		return self._value_at_decade(self.initial_sd, decade, self.sd_peak_growth)

	def new_stage_observation(self, stage):
		"""
			Calculates the new, Bayesian probability
		:param stage: int
		:return:
		"""

		# initial mu and sigma are given in Rui's paper - Mu == 4.42 and Sigma of 0.6
		# sigma/sqrt(25) is a given value from Jay in the probability equation - see photo from 2/27

		self.log.debug("New observation for stage {}".format(stage))

		decade = stage * constants.TIME_STEP_SIZE
		# 1 calculate the current stage mean and standard deviation based on the growth
		self.mean_at_stage[stage] = self.mean_at_decade(decade)
		self.sd_at_stage[stage] = self.sd_at_decade(decade)

	def calculate_scores(self, stage, mean_mean, mean_sd, sd_mean, sd_sd):
		# 2 calculate the Z scores - I think I'm doing this a bit wrong right now
		self.mean_z_scores[stage] = z_score(self.mean_at_stage[stage], mean_mean, mean_sd, sqrt_of_sample_size=1)
		self.sd_z_scores[stage] = z_score(observation=self.sd_at_stage[stage], mu=sd_mean, sigma=sd_sd, sqrt_of_sample_size=1)

		# 3 figure out the probability based on z score fit within discretized probability distribution
		self.mean_probabilities[stage] = self.get_probability(self.mean_z_scores[stage])
		self.sd_probabilities[stage] = self.get_probability(self.sd_z_scores[stage])

		self.bayesian_numerators[stage] = self.initial_probability * self.mean_probabilities[stage] * self.sd_probabilities[stage]
		self.log.info("Bayesian numerator at Stage {}: {}".format(self.name, stage, self.bayesian_numerators[stage]))

		# 4 Fit this probability into bayes' theorem - our new observed values will be
		# see kathy's jnotes - we'll want to come up with a way to store the probability
		# at each stage - we'll use those as the priors. We can then calculate all of this
		# up front, which we'll then use as multipliers when building our SDP.


def get_scenarios(number_of_stages=constants.NUMBER_TIME_STEPS):

	scenarios = []
	scenarios.append(Scenario("A", 0.2, 0, 0, sd_sd_growth=constants.SIGMA_OF_SIGMA, number_of_stages=number_of_stages))
	scenarios.append(Scenario("B", 0.2, 0, 0.05, sd_sd_growth=constants.SIGMA_OF_SIGMA, number_of_stages=number_of_stages))
	scenarios.append(Scenario("C", 0.2, 0, 0.10, sd_sd_growth=constants.SIGMA_OF_SIGMA, number_of_stages=number_of_stages))
	scenarios.append(Scenario("D", 0.2, 0.05, 0, sd_sd_growth=constants.SIGMA_OF_SIGMA, number_of_stages=number_of_stages))
	scenarios.append(Scenario("E", 0.1, 0.05, 0.05, sd_sd_growth=constants.SIGMA_OF_SIGMA, number_of_stages=number_of_stages))
	scenarios.append(Scenario("F", 0.1, 0.05, 0.10, sd_sd_growth=constants.SIGMA_OF_SIGMA, number_of_stages=number_of_stages))

	# Now calculate the sums of the numerators for each bayesian stage so we can make our denominator

	for stage in range(1, number_of_stages+1):
		means = []
		sds = []
		for scenario in scenarios:
			means.append(scenario.mean_at_stage[stage])
			sds.append(scenario.sd_at_stage[stage])

		for scenario in scenarios:
			scenario.calculate_scores(stage, numpy.mean(means), math.sqrt(stats.describe(means)[3]), numpy.mean(sds), math.sqrt(stats.describe(sds)[3]))

	denominators = [0]
	for stage in range(1, number_of_stages+1):
		stage_list = []
		for scenario in scenarios:
			stage_list.append(scenario.bayesian_numerators[stage])
		log.debug("Stage list: {}".format(stage_list))
		denominators.append(sum(stage_list))
		log.info("Denominator at stage {} is {}".format(stage, denominators[-1]))

	for stage in range(1, number_of_stages+1):
		log.debug("Getting Bayesian probabilities for stage {}".format(stage))
		for scenario in scenarios:
			scenario.bayesian_probabilities[stage] = float(scenario.bayesian_numerators[stage]) / denominators[stage]
			log.info("Scenario {}, stage {}: {}".format(scenario.name, stage, scenario.bayesian_probabilities[stage]))

	log.debug("MEANS")
	for scenario in scenarios:
		log.debug(scenario.mean_at_stage)

	log.debug("SDS")
	for scenario in scenarios:
		log.debug(scenario.sd_at_stage)

	log.debug("Mean Z-Scores")
	for scenario in scenarios:
		log.debug(scenario.mean_z_scores)

	log.debug("SD Z-Scores")
	for scenario in scenarios:
		log.debug(scenario.sd_z_scores)

	return scenarios


def present_value(value, year, discount_rate, compounding_rate=1):
	"""
		Calculates the present value of a future value similar to numpy.pv, except numpy.pv gave weird negative values
	:param value: The future value to get the present value of
	:param year: How many years away is it?
	:param discount_rate: The discount rate to use for all years/periods in between
	:param compounding_rate: How often during the period should values be compounded. 1 means once a year, 365 means daily, etc.
	:return:  present value of the provided value
	"""
	return value * (1 + float(discount_rate)/float(compounding_rate)) ** (-year*compounding_rate)


def building_and_maintenance_costs(initial_height, incremental_height, construction_cost_multiplier=constants.CONSTRUCTION_COST_MULTIPLIER):
	"""
	TODO: Might need to move the construction cost multiplier elsewhere

	:param initial_height:
	:param incremental_height:
	:param construction_cost_multiplier:
	:return:
	"""
	# build costs
	if initial_height == 0:  # when we don't have a levee, then our costs are different - we're not raising, we're doing initial construction
		build_cost = levee_construction_cost(incremental_height, build_year=0) * construction_cost_multiplier
	else:
		# assuming alterations don't have the cost multiplier because they have a fixed cost
		build_cost = levee_raise_cost(initial_height, incremental_height)

	return build_cost + MAINTENANCE_COST


def total_costs_of_choice(scenarios, initial_height, incremental_height, stage, observered_mean_flow, observed_flood_peak_variance):
	"""
		This will be our objective function for our DP

	:param scenarios:
	:param initial_height:
	:param incremental_height:
	:param stage: Helps us annualize it - if it's the last stage, then we do it differently because it's annualized FOREVER
	:return:
	"""

	levee_height = initial_height + incremental_height

	if levee_height > constants.MAXIMUM_LEVEE_HEIGHT:  # if this combination isn't allowed because it makes the levee too tall
		return constants.EXCLUSION_VALUE  # then return that it's just really expensive to raise it like this - fast if we just exclude it right off the bat

	# run the building and maintenance cost code ONCE, store the value
	cost = building_and_maintenance_costs(initial_height=initial_height, incremental_height=incremental_height)
	# for each scenario, get the costs of overtopping and failure for the new height multiplied by the bayesian probability

	###
	#  So, somewhere we want to calculate the expected value - To get the expected value of the flows, maybe something
	#  like the following:
	#                      stats.lognorm(*lognormal_transform(100,66)).expect()
	#
	#	Except, we can't just do the expected flow value. It should be probability of flows times a set of costs,
	#	so maybe we can't just use the expected value, but actually do need to discretize. Can we do this in a
	# 	speedy way? Maybe we build a table of levee heights x flows, where the values are calculated damages.
	#	Then, for speed, instead of calculating damages for every set of flows, we just look up the damages in
	#	the table by using bisect again. Since we'll probably see each levee height/flow combination many times.
	#	If we just look up each flow/levee height combination and multiply by the probabilities, that will help.
	#	We might even be able to just take a slice of flows for a levee height, and build a probability vector
	#	using a vectorized function where we get the z score of all of the values in the slice? Then we can just
	# 	multiply the values in the slice * the probabilities and sum. This could be a speedy approach.
	#
	###

	for scenario in scenarios:
		# TODO: Add flows and probabilities here. cost of overtopping
		cost += get_failure_costs(levee_height=levee_height, failure_function=vectorized_overtopping)  # TODO: Need to adjust this once we have the *probabilistic* flows

		# TODO: Add flows and probabilities here. cost of geotechnical failure
		cost += get_failure_costs(levee_height=levee_height, failure_function=levee_fails)

	# annualize it over the entire period

	return cost


def levee_raise_cost(current_height, incremental_height,):
	"""
		Cost of raising a levee incremental_height meters plus the fixed cost of alteration
	:param: current_height
	:param incremental_height:
	:return:
	"""

	changes = _levee_volume_change(current_height, incremental_height,
													length=constants.LEVEE_SYSTEM_LENGTH,
													slope=constants.LAND_SIDE_SLOPE,
													crown_width=constants.LEVEE_CROWN_WIDTH,
													number_of_sides=2,)
	cost = _levee_volume_cost(changes['volume_change'])  # get the cost of the change in volume
	cost += constants.FIXED_LEVEE_ALTERATION_COST  # add the cost of any construction project
	cost += changes['area_change'] * constants.LAND_PRICE  # add the cost of additional land that this levee sits on


def maintenance_cost_stage(length=constants.LEVEE_SYSTEM_LENGTH,
						   cost_per_length=constants.MAINTENANCE_COST_PER_METER,
						   period_length=constants.TIME_STEP_SIZE*10,
						   discount_rate=constants.DISCOUNT_RATE):
	"""
		Gets the maintenance cost for the entire stage, discounted to the beginning of the stage

		This only needs to be run once and then incorporated into all calculations with a levee height greater than 0
		since for every stage/height the value will be the same, and the DP will handle discounting it back to present day
		for stages past the first
	:param length:
	:param cost_per_length:
	:param period_length:
	:param discount_rate:
	:return:
	"""
	annual_cost = length * cost_per_length
	total_cost = annual_cost  # assign the base annual cost for year 0

	for year in range(1, period_length+1):
		total_cost += present_value(annual_cost, year, discount_rate)

	return total_cost


def _levee_volume_cost(volume,
					   year=0,
						material_cost=constants.COST_OF_SOIL,
						discount_rate=constants.DISCOUNT_RATE):
	"""
		TODO: We might not want to present value this because we'll do the present value adjustment once later
	:param volume:
	:param year:
	:param material_cost:
	:param discount_rate:
	:return:
	"""

	return volume * present_value(material_cost, year=year, discount_rate=discount_rate)


def _levee_volume(height,
					length=constants.LEVEE_SYSTEM_LENGTH,
					slope=constants.LAND_SIDE_SLOPE,
					crown_width=constants.LEVEE_CROWN_WIDTH,
					number_of_sides=2,):

	base_width = crown_width + (1/slope * height)
	xc_area = (crown_width+base_width)/2 * height
	return {'area': xc_area * number_of_sides, 'volume': xc_area * length * number_of_sides}


def _levee_volume_change(initial_height, incremental_height,
						 length=constants.LEVEE_SYSTEM_LENGTH,
						slope=constants.LAND_SIDE_SLOPE,
						crown_width=constants.LEVEE_CROWN_WIDTH,
						number_of_sides=2,):

	new_height = initial_height + incremental_height
	old_volume = _levee_volume(initial_height, length, slope, crown_width, number_of_sides)
	new_volume = _levee_volume(new_height, length, slope, crown_width, number_of_sides)

	return {'area_change': new_volume['area'] - old_volume['area'],
			'volume_change': new_volume['volume'] - old_volume['volume']
			}


def levee_construction_cost(height,
							build_year,
							length=constants.LEVEE_SYSTEM_LENGTH,
							slope=constants.LAND_SIDE_SLOPE,
							crown_width=constants.LEVEE_CROWN_WIDTH,
							number_of_sides=2,
							material_cost=constants.COST_OF_SOIL,
							discount_rate=constants.DISCOUNT_RATE):
	"""
		Calculates the cost of building a levee.

		TODO: NEED TO TAKE INTO ACCOUNT THE SLOPE UNDERNEATH AND HOW THAT REDUCES SOIL NEEDS.

	:param height: Height of levee above the floodplain - in meters
	:param build_year: when are we building the levee (in years from now). Used for discounting. Likely to be 0 since this is for initial builds
	:param length:  the length in meters of the levee to build
	:param slope:  the slope of the levee itself
	:param crown_width:  # how thick the top of the levee is
	:param number_of_sides:  values can be 1 or 2 - are we on a single side of the river or both sides?
	:param material_cost:  how much each cubic meter of soil costs, in dollars
	:param construction_multiplier:  How much should fixed costs be multiplied by to get true cost of construction
	:return:
	"""

	area_and_volume = _levee_volume(height, length, slope, crown_width, number_of_sides)
	cost = _levee_volume_cost(area_and_volume['volume'], build_year, material_cost, discount_rate)
	cost += area_and_volume['area'] * constants.LAND_PRICE  # add the cost of purchasing land to the construction
	return cost


# Flow corresponding to aNY specific water level (from bottom of the river), calculated by Manning's Equation
def get_flow_for_height(water_height):
	"""
		This is Rui's function for this exactly, incorporated here and adapted to my variable names
	:param water_height:
	:return: flow value for water reaching that height
	"""

	if water_height >= constants.TOE_HEIGHT:  # If water level is above the toe and below the top of the levee
		cross_section = constants.LEVEED_CHANNEL_WIDTH * constants.CHANNEL_DEPTH + (constants.LEVEED_CHANNEL_WIDTH + constants.LEVEED_CHANNEL_WIDTH_TO_TOE) * constants.FLOODPLAIN_HEIGHT / 2 + (2*constants.LEVEED_CHANNEL_WIDTH_TO_TOE + 2 * (water_height - constants.TOE_HEIGHT) / constants.WATER_SIDE_SLOPE) * (water_height - constants.TOE_HEIGHT) / 2
		# Cross section area of flow at water_height depth
		wetted_perimeter = constants.LEVEED_CHANNEL_WIDTH + 2 * constants.CHANNEL_DEPTH + 2 * math.sqrt(((constants.LEVEED_CHANNEL_WIDTH_TO_TOE - constants.LEVEED_CHANNEL_WIDTH) / 2) ** 2 + (constants.FLOODPLAIN_HEIGHT) ** 2) + 2 * math.sqrt(
			((water_height - constants.TOE_HEIGHT) / constants.WATER_SIDE_SLOPE) ** 2 + (water_height - constants.TOE_HEIGHT) ** 2)
		# Wetted perimeter
	else:
		if water_height >= constants.CHANNEL_DEPTH:  # If water level is above the channel depth and below the toe of the levee
			cross_section = constants.LEVEED_CHANNEL_WIDTH * constants.CHANNEL_DEPTH + (2 * constants.LEVEED_CHANNEL_WIDTH + 2 * (water_height - constants.CHANNEL_DEPTH) / constants.FLOODPLAIN_SLOPE) * (water_height - constants.CHANNEL_DEPTH) / 2
			# Cross section area of flow at water_height depth
			wetted_perimeter = constants.LEVEED_CHANNEL_WIDTH + 2 * constants.CHANNEL_DEPTH + 2 * math.sqrt(((water_height - constants.CHANNEL_DEPTH) / constants.FLOODPLAIN_SLOPE) ** 2 + (water_height - constants.CHANNEL_DEPTH) ** 2)
			# Wetted perimeter
		else:
			cross_section = water_height * constants.LEVEED_CHANNEL_WIDTH  # Cross section area of flow at water_height depth
			wetted_perimeter = 2 * water_height + constants.LEVEED_CHANNEL_WIDTH  # Wetted perimeter

	velocity = constants.MANNING_CONVERSION_FACTOR / constants.MANNINGS_N * (cross_section / wetted_perimeter) ** (2 / 3) * math.sqrt(constants.LONGITUDINAL_SLOPE_OF_CHANNEL)  # Water velocity
	overflow = velocity * cross_section  # Flow
	return overflow


def get_flow_height_index():
	flow_heights = {}
	number_of_steps = int((constants.MAXIMUM_LEVEE_HEIGHT - constants.INITIAL_LEVEE_HEIGHT)/constants.LEVEE_HEIGHT_INCREMENT)+1  # this gives us the number of increments for numpy.linspace
	for height in numpy.linspace(constants.INITIAL_LEVEE_HEIGHT, constants.MAXIMUM_LEVEE_HEIGHT, num=number_of_steps):
		input_height = height + constants.TOE_HEIGHT  # add the toe height so that our calculated heights are levee heights
		flow_at_height = get_flow_for_height(input_height)

		flow_heights[flow_at_height] = height  # this is specifically height and not input_height so we can compare it to levee heights

	return flow_heights

# DEFINE CONSTANTS BASED ON THESE FUNCTIONS
MAINTENANCE_COST = maintenance_cost_stage()  # make it a constant so we can just reference it
FLOW_HEIGHT_INDEX = get_flow_height_index()  # gives us the levee height of a specific flow
FLOW_HEIGHT_INDEX_KEYS = tuple(sorted(FLOW_HEIGHT_INDEX.keys()))


def get_required_levee_height(flow, flow_height_index=FLOW_HEIGHT_INDEX, flow_height_index_keys=FLOW_HEIGHT_INDEX_KEYS):
	"""

	:param flow:
	:param flow_height_index:
	:param flow_height_index_keys: We send in the keys as well so that we can sort them ONCE and not every time - it's for performance
	:return: required levee hight to contain given `flow` - if flow is infeasible, returns constants.MAXIMUM_LEVEE_HEIGHT + 1, so any value greater than MAXIMUM_LEVEE_HEIGHT should be caught
	"""
	if flow > flow_height_index_keys[-1]:  # if the flow is greater than the highest option, return max height + 1
		return constants.MAXIMUM_LEVEE_HEIGHT + 1  # this is infeasible and we'll catch it in anything that calls this

	corresponding_height_index = bisect.bisect_left(flow_height_index_keys, flow)
	corresponding_flow = flow_height_index_keys[corresponding_height_index]
	return flow_height_index[corresponding_flow]

vectorized_get_required_levee_height = numpy.vectorize(get_required_levee_height)

def levee_is_overtopped(flow, levee_height):
	"""
		Tells us if a levee of height levee_height is overtopped by a flow of magnitude flow.

		We might not use this, instead opting for a faster numpy approach
	:param flow:
	:param levee_height:
	:return:
	"""

	if get_required_levee_height(flow) > levee_height:
		return True
	else:
		return False


vectorized_overtopping = numpy.vectorize(levee_is_overtopped)


def levee_fails(flows, levee_height, failure_scaling_factor=constants.FAILURE_SCALING_FACTOR):
	"""
		Determines if a levee fails based on a linear probability of failure from 0 at the bottom to 1 at the top, according
		to Hui et al, 2018. But, that estimate seems high, so we reduce that probability by multiplying it by
		the failure_scaling_factor, then randomly assessing if the levee fails.

		TODO: We might not want to do it this way because the sample size might not be large enough like a Monte Carlo.
		TODO: Instead, we might want to just multiply the failure chance times the cost and assume it's deterministic.

		TODO: This function could be made faster if it was combined with the overtopping function and they returned one set of damages
	:param flows:
	:param levee_height:
	:param failure_scaling_factor:
	:return:
	"""

	water_heights = vectorized_get_required_levee_height(flows)
	failure_chance = (water_heights / levee_height) * failure_scaling_factor
	failure_values = numpy.random.random_sample(flows.size)
	failure_values[water_heights > levee_height] = 1  # basically, this allows us to pass in all flows, and it automatically excludes any flows that overtop the levee because we calculate those damages elsewhere - we could do it here instead and increase the speed a bit, with some refactoring

	return failure_values < failure_chance  # returns True for failure values smaller than the failure chance and False for ones larger


def get_failure_costs(flows, levee_height, probabilities, failure_function=vectorized_overtopping):
	"""

	:param flows: a numpy array of all of the flows from the discretized distribution
	:param levee_height: the height of the levee these flows are running near
	:param probabilities: a numpy array of the probabilities of that flow
	:param failure_function: a function, ideally numpy vectorized, that returns a boolean array indicating True if a failure
			occurs with said flow, and False if no failure occurs
	:return: total cost across all probabilistic values of overtopping
	"""

	overtopped = failure_function(flows, levee_height)  # gets if a levee of this height is overtopped by this flow
	costs = numpy.zeros_like(flows)  # make a cost array with 0 for each flow
	numpy.putmask(costs, mask=overtopped, values=constants.FLOOD_DAMAGE_COST_FOR_EACH_FAILURE)

	return numpy.sum(costs * probabilities)  # the summed cost of each times its probability is our overtopping cost


def z_score(observation, mu, sigma, sqrt_of_sample_size=constants.SQRT_INITIAL_SAMPLE_SIZE):
	"""
		Just a simple equation to calculate z-scores
	:param observation:
	:param mu:
	:param sigma:
	:return:
	"""
	return float(observation - mu)/float(sigma/sqrt_of_sample_size)

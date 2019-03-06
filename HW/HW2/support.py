import math
import bisect  # we'll use this to find appropriate z_score probabilities

import numpy
from scipy import stats

from . import constants


class Scenario(object):
	def __init__(self, name,
				 initial_probability,
				 mean_peak_growth,
				 sd_peak_growth,
				 sd_sd_growth=constants.SIGMA_OF_SIGMA,
				 initial_mean=constants.INITIAL_MEAN_OF_ANNUAL_FLOOD_FLOW,
				 initial_sd=constants.INITIAL_SD_OF_ANNUAL_FLOOD_FLOW):

		self.name = name
		self.initial_probability = initial_probability
		self.mean_peak_growth = mean_peak_growth  # mean peak flood growth by decade
		self.sd_peak_growth = sd_peak_growth  # standard deviation of peak flood growth by decade
		self.sd_sd_growth = sd_sd_growth  # growth of the standard deviation itself

		# these are the annual flow paremeters in the initial period, but *NOT* the
		# parameters for the PEAK flows. We use these to construct a lognormal
		# distribution, to then get the peak flows out of it. The flows are lognormal
		# but the probability distributions (for the z scores when updating??) are
		# normal. This is something that needs confirming. See page 6 of Rui's paper
		self.initial_mean = initial_mean
		self.initial_sd = initial_sd

		self.prior_probabilities = [initial_probability]
		self.all_observations = []  # we'll store all observations we've seen here, so we can use it to update

		self.probability_distribution_discretization = numpy.linspace(start=constants.PROBABILITY_DISTRIBUTION_LIMITS[0],
																	  stop=constants.PROBABILITY_DISTRIBUTION_LIMITS[1],
																	  num=constants.PROBABILITY_DISTRIBUTION_DISCRETIZATION_UNITS)
		self.probabilities = {}  # we'll index and store total probabilities here for each item in the probability discretization

		self._index_z_probabilities()

	def _index_z_probabilities(self):
		total_items = len(self.probability_distribution_discretization)
		for i, lower_z in enumerate(self.probability_distribution_discretization):
			if i+1 == total_items:  # don't process the last item - it's the top of a range
				break

			upper_z = self.probability_distribution_discretization[i+1]
			lower_z_cum_prob = stats.norm.cdf(lower_z)
			upper_z_cum_prob = stats.norm.cdf(upper_z)
			self.probabilities[lower_z] = upper_z_cum_prob - lower_z_cum_prob

	def get_probability(self, z_score):

		# make sure it's in range - if it's greater than the max, use the max. If it's less than the min, use the min
		z_score = min(z_score, constants.PROBABILITY_DISTRIBUTION_LIMITS[1])
		z_score = max(z_score, constants.PROBABILITY_DISTRIBUTION_LIMITS[0])

		probability_index = bisect.bisect_right(self.probability_distribution_discretization, z_score) - 1 # get the location of the min z_score involved
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

	def update(self, new_observation, stage):
		"""
			Calculates the new, Bayesian probability
		:param prior_probability:
		:param new_observation:
		:param all_observations:
		:return:
		"""

		# initial mu and sigma are given in Rui's paper - Mu == 4.42 and Sigma of 0.6
		# sigma/sqrt(25) is a given value from Jay in the probability equation - see photo from 2/27

		decade = stage * constants.TIME_STEP_SIZE
		# 1 calculate the current stage mean and standard deviation based on the growth
		current_mean = self.mean_at_decade(decade)
		current_sd = self.sd_at_decade(decade)

		# 2 calculate the Z score
		z_score = float(new_observation - current_mean) / float(current_sd / constants.SQRT_INITIAL_SAMPLE_SIZE)

		# 3 figure out the probability based on z score fit within discretized probability distribution
		probability = self.get_probability(z_score)

		# 4 Fit this probability into bayes' theorem - our new observed values will be
		# see kathy's jnotes - we'll want to come up with a way to store the probability
		# at each stage - we'll use those as the priors. We can then calculate all of this
		# up front, which we'll then use as multipliers when building our SDP.

		# this next block is wrong, just keeping it for now
		self.all_observations.append(new_observation)
		self.prior_probabilities[stage] = float(self.prior_probabilities[stage-1] * new_observation) / sum(self.all_observations)


def get_scenarios():
	scenarios = []
	scenarios.append(Scenario("A", 0.2, 0, 0, sd_sd_growth=constants.SIGMA_OF_SIGMA))
	scenarios.append(Scenario("B", 0.2, 0, 0.05, sd_sd_growth=constants.SIGMA_OF_SIGMA))
	scenarios.append(Scenario("C", 0.2, 0, 0.10, sd_sd_growth=constants.SIGMA_OF_SIGMA))
	scenarios.append(Scenario("D", 0.2, 0.05, 0, sd_sd_growth=constants.SIGMA_OF_SIGMA))
	scenarios.append(Scenario("E", 0.1, 0.05, 0.05, sd_sd_growth=constants.SIGMA_OF_SIGMA))
	scenarios.append(Scenario("F", 0.1, 0.05, 0.10, sd_sd_growth=constants.SIGMA_OF_SIGMA))

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


def levee_overtopping_cost():
	pass


def levee_failure_cost():
	pass


def levee_construction_cost(height,
							build_year,
							length=constants.LEVEE_SYSTEM_LENGTH,
							slope=constants.LAND_SIDE_SLOPE,
							crown_width=constants.LEVEE_CROWN_WIDTH,
							number_of_sides=2,
							material_cost=constants.COST_OF_SOIL,
							construction_multiplier=constants.CONSTRUCTION_COST_MULTIPLIER,
							discount_rate=constants.DISCOUNT_RATE):
	"""
		Calculates the cost of building a levee.

		TODO: NEED TO TAKE INTO ACCOUNT THE SLOPE UNDERNEATH AND HOW THAT REDUCES SOIL NEEDS.

	:param height: Height of levee above the floodplain - in meters
	:param build_year: when are we building the levee (in years from now). Used for discounting
	:param length:  the length in meters of the levee to build
	:param slope:  the slope of the levee itself
	:param crown_width:  # how thick the top of the levee is
	:param number_of_sides:  values can be 1 or 2 - are we on a single side of the river or both sides?
	:param material_cost:  how much each cubic meter of soil costs, in dollars
	:param construction_multiplier:  How much should fixed costs be multiplied by to get true cost of construction
	:return:
	"""
	base_width = crown_width + (1/slope * height)
	xc_area = (crown_width+base_width)/2 * height
	volume = xc_area * length & number_of_sides
	cost = volume * present_value(material_cost*construction_multiplier, year=build_year, discount_rate=discount_rate)

	return cost


def z_score(observation, mu, sigma):
	"""
		Just a simple equation to calculate z-scores
	:param observation:
	:param mu:
	:param sigma:
	:return:
	"""
	return float(observation - mu)/sigma

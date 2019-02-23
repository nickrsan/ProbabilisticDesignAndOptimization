"""
	A Monte-Carlo approach to Jay Lund's Frat House Burning problem. Provides cost distributions for choosing
	to build a normal or fire-resistant house, based on each one's probability of burning.

"""

import random
import numpy
import math
from scipy import stats


class Option(object):
	def __init__(self, name, probability_of_burning, cost):
		self.name = name
		self.probability_of_burning = probability_of_burning
		self.cost = cost

		self.results = []  # each item in the list is a total cost for an iteration, which will be summed by year


options = (
	Option(name="normal", probability_of_burning=0.1, cost=150000),
	Option(name="fire resistant", probability_of_burning=0.05, cost=300000)
)


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


def full_monty(iterations=100000, build_options=options, years=150, discount_rate=0.05):
	"""
		Runs a Monte Carlo simulation comprised of the specified number of iterations, where each iteration
		goes through `years` years. Each year has a single random value - when that random value is less
		than each option's probability of fire, that house burns. So, for a very low random value, both
		houses burn in that same year. This was chosen because, in evaluating them as alternatives,
		we should think of each house as a standin for each other, so we can take the random values as
		events of specific extremity that can cause fire, where the resistant house avoids some of the
		fires of the normal house.
	:param iterations:
	:param build_options:
	:param years:
	:param discount_rate:
	:return:
	"""
	for iteration in range(iterations):
		if iteration % 1000 == 0:  # every 1000 iterations, let's print something to show where we're at
			print(iteration)

		for option in build_options:
			option.results.append(option.cost)  # for each iteration, we start with the cost to initially build the place

		for year in range(1, years):
			burn = random.random()  # get the burn value - this is random from 0-1, but doesn't say if it burned yet
			for option in build_options:
				if burn < option.probability_of_burning:  # this means it burned!
					option.results[iteration] += present_value(option.cost, year=year, discount_rate=discount_rate)  # add the present value cost of it burning that year

	print("Complete")
	for option in options:
		print(option.name)
		option_stats = stats.describe(numpy.array(option.results))
		print(option_stats)
		print("standard deviation: {}".format(math.sqrt(option_stats[3])))


if __name__ == "__main__":
	random.seed(20190131)
	full_monty()

"""
	A Monte-Carlo approach to Jay Lund's Frat House Burning problem. Provides cost distributions for choosing
	to build a normal or fire-resistant house, based on each one's probability of burning.

"""

import numpy
import math
from scipy import stats

import seaborn
from matplotlib import pyplot

def lognormal_transform(mu, sigma):
	"""
		Given mu and sigma values of a lognormal distribution, returns mu and sigma of the underlying
		normal distribution that can be used as inputs for numpy.random.lognormal
	:param mu:
	:param sigma:
	:return:
	"""

	variance_lognormal = numpy.log((float(sigma**2)/float(mu**2))+1)
	sigma_lognormal = math.sqrt(variance_lognormal)
	mu_lognormal = numpy.log(mu) - (variance_lognormal/2)

	return mu_lognormal, sigma_lognormal


class Filter(object):
	def __init__(self, removal_mean=0.9, removal_sd=0.05, max_removal=1.0, cost=100000):

		transformed_mean, transformed_sd = lognormal_transform(removal_mean, removal_sd)

		self.removal_mean = transformed_mean
		self.removal_sd = transformed_sd
		self.max_removal = max_removal
		self.removal_efficiency = None
		self.cost = cost

	def filter_water(self, contamination_concentration):
		"""
			Given a contamination_concentration, removes contaminants randomly from a log-normal distribution - max
			contaminant removal is 1. Returns the *new* contaminant level, not the amount removed
		:param contamination_concentration:  int or float of contamination concentration
		:return:  contamination concentration post filter
		"""
		removal_proportion = numpy.random.lognormal(self.removal_mean, self.removal_sd)
		removal_proportion = min(removal_proportion, self.max_removal)  # can't remove more than max_removal, so take whichever is smaller between it and the random value
		removal_proportion = max(removal_proportion, 0)  # can't be less than 0, so take the larger of 0 and the actual amount
		self.removal_efficiency = removal_proportion
		return contamination_concentration * (1-removal_proportion)


class FilterSeries(object):
	"""
		Manages filter objects in a series
	"""

	def __init__(self, filter_cost=100000, max_cost=1000000):
		"""
			Creates and manages filters in a series, based on the cost of the filter and the max_cost.
			Eg, with a filter cost of 100 and a max cost of 1000, creates 10 filters. Max cost is set
			based on the cost of the alternative to filtering
		:param filter_cost: How much does a single filter cost?
		:param max_cost: How much does our alternative to filtering cost?
		"""
		self.filters = []
		self.outbreaks = {}
		self.contaminant_levels = {}
		self.total_cost = []
		self.removal_efficiency = []

		self.filter_cost = filter_cost
		self.max_cost = max_cost
		for _ in range(0, max_cost, filter_cost):  # steps from 0 to max cost by units of filter cost
			self.filters.append(Filter(cost=filter_cost))

	def reset(self):
		"""
			Gets it ready for another run of filtering by clearing outputs
		:return:
		"""
		self.contaminant_levels = {}  # stores contaminant levels by cost, essentially, for the number of filters
		self.outbreaks = {}
		self.total_cost = []
		self.removal_efficiency = [0]  # start with removal efficiency at 1 so we can just multiply later

	def filter_water(self, contamination_level):
		"""
			Runs the filtering pipeline in series, storing the contaminant levels by the cost for that length of filter
			seties. Returns a dict of costs
		:return:
		"""

		self.contaminant_levels['0'] = contamination_level  # put the amount of base contamination with no reduction here
		self.outbreaks['0'] = is_outbreak(contamination_level)
		self.total_cost.append(1000000 if self.outbreaks['0'] is True else 0)

		cost = 0  # start with 0 cost
		for i, each_filter in enumerate(self.filters):
			cost += each_filter.cost  # bump it with each filter in the series
			contamination_level = each_filter.filter_water(contamination_concentration=contamination_level)

			# figure out what proportion was previously removed and what's left now to remove
			removed_contaminants = self.removal_efficiency[i]  # use i because i is one less than the actual index since we have 1 filter to start in the loop
			remaining_contaminants = 1 - removed_contaminants
			# new removal efficiency == the amount previously removed plus the new amount removed
			self.removal_efficiency.append(removed_contaminants + (each_filter.removal_efficiency * remaining_contaminants))

			self.contaminant_levels[str(cost)] = contamination_level
			self.outbreaks[str(cost)] = is_outbreak(contamination_level)
			self.total_cost.append(cost + (1000000 if self.outbreaks[str(cost)] is True else 0))


def is_outbreak(contaminant_level, minimum_outbreak_level=0.05, maximum_outbreak_level=2.0):
	"""
		Given a final contaminant level returns True if that level would trigger an outbreak and False if it doesn't.
		Uses a linear random value to determine if an outbreak has occurred.

		Kathy and Patrick noted that instead of checking for an outbreak in each instance (closer to a simulation), could
		also get the probability distribution for the results and use that in conjunction with the probability of
		outbreaks for concentrations to add an expected cost for each level of filtration.

		My method is easier for me to conceptualize, so I'm going with that.
	:param contaminant_level:
	:param minimum_outbreak_level: minimum level at which an outbreak could occur
	:param maximum_outbreak_level: maximum level where an outbreak will always occur
	:return:
	"""
	if contaminant_level < minimum_outbreak_level:
		return False
	if contaminant_level > maximum_outbreak_level:
		return True

	outbreak_probability = (contaminant_level - minimum_outbreak_level) / maximum_outbreak_level  # subtract out the min value then divide by the max value so that it's normalized to a max value of one
	outbreak_rand = numpy.random.random()

	if outbreak_rand < outbreak_probability:  # then we have an outbreak!
		return True
	else:
		return False


def full_monty(iterations=100000, filter_cost=100000, max_cost=1000000, contamination_mean=5, contamination_sd=3):
	"""
		Run a monte carlo distribution for the filtering problem
	:param iterations:
	:param filter_cost:
	:param max_cost:
	:return:
	"""

	filter_series = FilterSeries(filter_cost=filter_cost, max_cost=max_cost)  # create a full filter series.

	filter_results = {}
	outbreaks = {}
	total_costs = [[], [], [], [], [], [], [], [], [], [], []]  # 0-10 filters, storing total resulting cost in a list
	removal_efficiencies = [[], [], [], [], [], [], [], [], [], [], []]
	for cost in range(0, max_cost+1, filter_cost):  # for every possible filter cost set, make an empty list of contaminant values
		filter_results[str(cost)] = []

	starting_contamination_mean, starting_contamination_sd = lognormal_transform(contamination_mean, contamination_sd)

	for cost in range(0, max_cost+1, filter_cost):
		outbreaks[str(cost)] = []  # set up the outbreak results

	for iteration in range(iterations):
		if iteration % 1000 == 0:  # every 1000 iterations, let's print something to show where we're at
			print(iteration)

		contamination_level = numpy.random.lognormal(mean=starting_contamination_mean, sigma=starting_contamination_sd)

		filter_series.reset()  # clear the results
		filter_series.filter_water(contamination_level)

		for cost in filter_series.contaminant_levels:
			filter_results[str(cost)].append(filter_series.contaminant_levels[str(cost)])
			outbreaks[str(cost)].append(filter_series.outbreaks[str(cost)])
			for i, tc in enumerate(filter_series.total_cost):
				total_costs[i].append(tc)
			for i, efficiency in enumerate(filter_series.removal_efficiency):
				removal_efficiencies[i].append(efficiency)

	print("Filter phase complete. Numbers below are *just* costs for filtration, not outbreak costs")
	for cost in range(0, max_cost+1, filter_cost):  # for every possible filter cost set, make an empty list of contaminant values
		print("\nCost Level {}".format(cost))
		base_cost_stats = stats.describe(numpy.array(filter_results[str(cost)]))
		print(base_cost_stats)
		print("standard deviation: {}".format(math.sqrt(base_cost_stats[3])))
		print("number of outbreaks: {}".format(numpy.array(outbreaks[str(cost)]).sum()))

	cost_means = []
	print("Cost stats when including cost of outbreak")
	for num_filters in range(int(max_cost/filter_cost)+1):
		print("{} filters".format(num_filters))
		costs = total_costs[num_filters]
		cost_stats = stats.describe(numpy.array(costs))
		cost_means.append(int(cost_stats[2]/1000))  # divide by 1000 because we'll plot in thousands
		print(cost_stats)
		print("standard deviation: {}".format(math.sqrt(cost_stats[3])))

	plot(cost_means, len(cost_means), nsamples=iterations)
	plot_efficiencies(removal_efficiencies, nsamples=iterations)

	print("Efficiency Stats")
	for num_filters, series in enumerate(removal_efficiencies[1:4]):
		print("{} filters".format(num_filters+1))
		efficiency_stats = stats.describe(numpy.array(series))
		print(efficiency_stats)
		print("standard deviation: {}".format(math.sqrt(efficiency_stats[3])))


def plot(cost_means, x_size=11, nsamples=None):
	seaborn.set()  # style="whitegrid")

	print("Plotting")
	current_plot_ax = seaborn.lineplot(x=range(x_size), y=cost_means, markers=True, )
	seaborn.scatterplot(x=range(x_size), y=cost_means, ax=current_plot_ax)
	current_plot_ax.set(ylim=(0, max(cost_means)),
						xlim=(0, x_size-1),
						xlabel="Number of filters",
						ylabel="Total cost including cost of outbreaks (thousand dollars)")
	current_plot_ax.set_title("Total Expected Cost of Contaminant Removal and Outbreaks with Filters in Series (nsamples={})".format(nsamples))
	# label points - via https://stackoverflow.com/a/37115496/587938
	ymin, ymax = current_plot_ax.get_ylim()
	y_offset = (ymax - ymin) / 20
	[current_plot_ax.text(p[0] - 0.15, p[1] - y_offset, p[1], color='b') for p in zip(range(x_size), cost_means)]

	pyplot.show()


def plot_efficiencies(removal_efficiencies, nsamples):

	seaborn.set()  # style="whitegrid")
	for i, series in enumerate(removal_efficiencies[1:4]):
		current_plot_ax = seaborn.distplot(series, hist=False, label="{} Filter{}".format(i+1, "s" if i+1 != 1 else ""))

	current_plot_ax.set(xlim=(0.70, 1),
						ylim=(0, 150),
						xlabel="Filter Removal Efficiency for N filters",
						ylabel="Probability Density",
						)#ylabel="Total cost including cost of outbreaks (thousand dollars)")
	current_plot_ax.set_title("Probability Density for Cumulative Removal Efficiency of a Series of Filters (nsamples={})".format(len(series)))
	pyplot.legend()
	pyplot.show()



if __name__ == "__main__":
	numpy.random.seed(20190203)
	full_monty()

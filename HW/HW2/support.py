from . import constants


class Scenario(object):
	def __init__(self, name, initial_probability, mean_peak_growth, sd_peak_growth):
		self.name = name
		self.initial_probability = initial_probability
		self.mean_peak_growth = mean_peak_growth  # mean peak flood growth by decade
		self.sd_peak_growrth = sd_peak_growth  # standard deviation of peak flood growth by decade

	def update(self):
		"""
			Some sort of Bayesian update code here
		:return:
		"""

def get_scenarios():
	scenarios = []
	scenarios.append(Scenario("A", 0.2, 0, 0))
	scenarios.append(Scenario("B", 0.2, 0, 0.05))
	scenarios.append(Scenario("C", 0.2, 0, 0.10))
	scenarios.append(Scenario("D", 0.2, 0.05, 0))
	scenarios.append(Scenario("E", 0.1, 0.05, 0.05))
	scenarios.append(Scenario("F", 0.1, 0.05, 0.10))

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



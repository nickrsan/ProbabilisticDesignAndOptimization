"""
	Constants defined in Hui et al 2018

	Using both original name here, but also a descriptive name with the same value.
	The purpose being that I'd *like* to use the descriptive name for it, but also
	want to be able to quickly translate equations as needed
"""
import math
import logging
import random
import numpy

log = logging.getLogger("levee.constants")

# set random seeds
random.seed(20190309)
numpy.random.seed(20190309)

EXCLUSION_VALUE = 9223372036854775808  # max value for a signed 64 bit int - this should force it to not be selected in minimization

# When getting probabilities from z scores during bayesian updating, how far out on the tails should we go before capping it, and how much should we group similar areas?
PROBABILITY_DISTRIBUTION_LIMITS = [-3, 3]
PROBABILITY_DISTRIBUTION_DISCRETIZATION_UNITS = 30  # how many blocks should we break the probability distribution up into for calculating probabilities from z scores

TIME_STEP_SIZE = 4  # decades - how often do we make a new decision about levee heights?
TIME_HORIZON = 20  # decades - how far out do we want to look in making decisions?
NUMBER_TIME_STEPS = int(TIME_HORIZON/TIME_STEP_SIZE)

Rt = 0.035
DISCOUNT_RATE = Rt  # inflation adjusted

L = 2000   # meters
LEVEE_SYSTEM_LENGTH = L

W = 90  # meters
LEVEED_CHANNEL_WIDTH_TO_TOE = W  # leveed channel total width till the toe of the levee

B = 60  # meters
LEVEED_CHANNEL_WIDTH = B

D = 1  # meter
CHANNEL_DEPTH = D

TANA = 0.25  # 1/4
TAN_ALPHA = TANA  # land side slope
LAND_SIDE_SLOPE = TANA

TANB = 0.5  # 1/2
TAN_BETA = TANB
WATER_SIDE_SLOPE = TANB

T = 0.01  # floodplain slope
TAU = T
FLOODPLAIN_SLOPE = T

Sc = 0.0005
LONGITUDINAL_SLOPE_OF_CHANNEL = Sc

Nc = 0.05  # mannings coefficient
MANNINGS_N = Nc  # roughness factor, basically
MANNING_CONVERSION_FACTOR = 1.0   # a conversion factor - would be 1.4859 if it's English Units (according to Rui)

Bc = 10  # meters
LEVEE_CROWN_WIDTH = Bc

# help us construct the flow distributions
mu_0s = 100  # m^3/s
INITIAL_MEAN_OF_ANNUAL_FLOOD_FLOW = mu_0s  # see note in paper about this value
sigma_0s = 66  # m^3/s
INITIAL_SD_OF_ANNUAL_FLOOD_FLOW = sigma_0s
INITIAL_SAMPLE_SIZE = 25  # given by Jay as the size fo the observations leading to initial mean and SD
SQRT_INITIAL_SAMPLE_SIZE = math.sqrt(INITIAL_SAMPLE_SIZE)  # just compute it once since we'll use this a lot more than the sample size itself
SIGMA_OF_SIGMA = 10  # given from Jay - assumption we'll make - growth of the standard deviation

DC = 10000000  # dollars
FLOOD_DAMAGE_COST_FOR_EACH_FAILURE = DC
FAILURE_SCALING_FACTOR = 0.05  # Rui has failurs occur linearly from 0 at the bottom to 1 at the top. This seems very high - 1/4 of levees that are 1/4 way up don't fail. This value scales those chances down

UC = 1  # $/m^2
LAND_PRICE = UC
C_SOIL = 30  # $/m^3
COST_OF_SOIL = C_SOIL
C_ADJUST = 1.3
CONSTRUCTION_COST_MULTIPLIER = C_ADJUST  # soft cost multiplier for construction management

H0 = 0   # meters
INITIAL_LEVEE_HEIGHT = H0

DISCRETIZED_DH = 0.1  # meters
LEVEE_HEIGHT_INCREMENT = DISCRETIZED_DH  # what level of change are we allowed to make to a levee?

H_max = 15  # meters
MAXIMUM_LEVEE_HEIGHT = H_max

DH_max = 10  # meters
MAXIMUM_UPGRADE_LEVEE_HEIGHT = DH_max

OM = 74000  # maintenance cost in dollars per kilometer
MAINTENANCE_COST_PER_METER = OM/1000

FLAC = 500000
FIXED_LEVEE_ALTERATION_COST = FLAC  # base cost for any change to levee

## CALCULATED CONSTANTS

# These two are values for the height at the base of the floodplain and the height at the toe of the levee - flows below
# the toe height never overflow
FLOODPLAIN_HEIGHT = (LEVEED_CHANNEL_WIDTH_TO_TOE - LEVEED_CHANNEL_WIDTH) * FLOODPLAIN_SLOPE  # Floodplain height
TOE_HEIGHT = CHANNEL_DEPTH + FLOODPLAIN_HEIGHT  # Water level at the toe of the levee
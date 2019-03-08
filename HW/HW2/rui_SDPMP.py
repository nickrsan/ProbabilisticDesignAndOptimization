#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

start = time.time()

import numpy as np
import math  # This will import math module
from scipy.stats import lognorm, norm
# A lognormal continuous random variable. logpdf(x, s, loc=0, scale=1), s is shape parameter logsigma, loc is locale parameter logmu, scale parameter of lognormal is exp(logmu)
# A normal continuous random variable. pdf(x, loc=0, scale=1), cdf(x, loc=0, scale=1)

from scipy.integrate import quad, simps
import csv
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

largenumber = 1e+20  # a very large number for calculation purpose

# Planning time periods
n = 200  # a levee total lifetime (yrs)

T = 40  # the year interval that a levee to be upgrading (yrs)
NT = int(n / T)  # the number of intervals in a levee's lifetime

# Levee design standard
H0 = 0.0  # The initial levee height (meter)
DELTAH = 0.1  # Step size of levee height (meter)
Hmin = 0.0  # The minimum levee height (meter)
Hmax = 12.0  # The maximum levee height (meter)
Hupmax = 10.0  # The maximum upgrading levee height (meter)

# Financial Parameters
DC = 10.0  # Flood Damage Cost, assuming simultaneous failures on two riversides with a constant value (million $)
Cland = 1.0  # Price of land ($/m**2)
Csoil = 30.0  # Soil compaction cost or construction cost per unit levee material ($/m**3)
Cadjust = 1.3  # Soft Cost Multiplier considering management
R = 0.035  # Real (inflation-adjusted) discount rate

# Stationary Annual Flow Parameters
mu0 = 100.0  # Mean annual flow (m**3/s), the bigger the lower peak probability and longer tail
sigma0 = 66.0  # Standard deviation of annual flow
cv0 = sigma0 / mu0  # Coefficient of variation
var0 = sigma0 ** 2  # Variation of flow, the bigger the lower peak probability and longer tail
logmu0 = math.log((mu0 ** 2) / math.sqrt(var0 + mu0 ** 2))
# location parameter of annual flow # the bigger the lower peak probability and longer tail
logsigma0 = math.sqrt(math.log(var0 / (mu0 ** 2) + 1))
# scale parameter of annual flow  # the bigger the lower peak probability and longer tail
expmu0 = math.exp(logmu0)

deltaq = 0.1  # integration interval

# climate scenarios
NMU = 3  # number of different changing mean
NSIGMA = 2  # number of different changing standard deviation
NA = NMU * NSIGMA  # number of climate scenarios

mus = 0.5
sigmas = 1.0

muslope = np.zeros(NA)
# initialization of variation of possible location parameter of annual flow
for i in range(NA):
	# ki = k % NMU
	# index for changing mean
	muslope[i] = mus / 100 * i
sigmaslope = np.zeros(NA)
# initialization of additional variation of possible scale parameter of annual flow
for i in range(NA):
	# ik = k / NMU
	# index for changing variation
	sigmaslope[i] = sigmas / 100 * i

PA = np.zeros(NA)
# initialization of probability of each possible climate scenarios, the same
MU = np.zeros((n, NA))
# initialization of mean annual flow for n years and NA climate scenarios
SIGMA = np.zeros((n, NA))
# initialization of standard deviation of annual flow for n years and NA climate scenarios
LOGMU = np.zeros((n, NA))
# initialization of location parameter of annual flow for n years and NA climate scenarios
LOGSIGMA = np.zeros((n, NA))
# initialization of scale parameter of annual flow for n years and NA climate scenarios
for i in range(n):
	# stage or time i
	for k in range(NA):
		# a climate scenario or a combination of mean annual flow MU and standard deviation SIGMA
		PA[k] = (1.0 / NA)
		MU[i, k] = mu0 + muslope[k] * mu0 * (i + 1)
		# annual average flow for time i and climate scenario j, changing linearly
		SIGMA[i, k] = sigma0 + sigmaslope[k] * sigma0 * (i + 1)
		# standard deviation of annual flow for time i and climate scenario j, changing linearly
		cv = SIGMA[i, k] / MU[i, k]
		# additional change to Coefficient of variation, not necessary since sigma will change with mu anyway
		var = SIGMA[i, k] ** 2
		# Variation of flow, the bigger the lower peak probability and longer tail
		LOGMU[i, k] = math.log(((MU[i, k]) ** 2) / math.sqrt(var + (MU[i, k]) ** 2))
		# location parameter of annual flow for time i and climate scenario j
		LOGSIGMA[i, k] = math.sqrt(math.log(var / ((MU[i, k]) ** 2) + 1))
		# scale parameter of annual flow for time i and climate scenario j
	# print i+1, k+1, MU[i,k], SIGMA[i,k], cv, var, LOGMU[i,k], LOGSIGMA[i,k]

# Given Channel Parameters
Sc = 0.0005  # Longitudinal slope of the Channel, also the slope of the hydraulic grade line or the linear hydraulic head loss
Nc = 0.05  # Roughness factor of the channel section, also known as the Manning coefficient
kconversion = 1.0  # A conversion factor k = 1.4859 for English Unit

# Levee geometry parameters
W = 90.0  # Total width till the toe of the levee (meter)
B = 60.0  # Total Channel width (meter)
D = 1.0  # Channel Depth (meter)
WS = 1.0 / 2.0  # Water side-slopes tan(waterside angle)= 1:2
LS = 1.0 / 4.0  # Land side-slopes tan(langdside angle)= 1:4
FS = 0.01  # Floodplain slope
Bc = 10.0  # Levee crown width (meter)
L = 2000.0  # Total levee length (meter)

# OPERATION AND MAINTENANCE COST
OM = 74000  # $/km
AOM = OM * (L / 1000) / 1000000  # annual operation and maintenance cost

# A fixed levee alteration cost when the levee system is upgraded to represent the costs for extra land acquisition, legal services, design and consulting, etc.
FLAC = 0.5  # million$

# Levee Design parameters
NH = int(round((Hmax - Hmin) / DELTAH + 1))  # Number of discretized initial levee height
NHUP = int(round((Hupmax) / DELTAH + 1))  # Number of discretized upgrading levee height

EXH = np.zeros((n, NH))
# initialization of the discretized existing levee height as state
# (EXH[i][l]) i =1:n stage or time; l =1:NH discretized existing levee height
for i in range(n):
	# stage or time i
	for l in range(NH):
		# index for existing levee height l
		(EXH[i][l]) = DELTAH * l  # Assign all possible existing heights

UPH = np.zeros((n, NHUP))
# initialization of the discretized upgrading levee height
# (UPH[i][m]) i =1:n stage or time;  m =1:NHUP discretized upgrading levee height
for i in range(n):
	# stage or time i
	for m in range(NHUP):
		# index for upgrading levee height m
		if (i % T) == 0:
			# annual expected total cost at upgrading time period every T year interl, including expected damage cost and construction cost
			(UPH[i][m]) = DELTAH * m  # Assign all possible upgrading heights
		else:
			# annual expected total cost at none upgrading time periods, upgradh = 0, only expected damage cost
			(UPH[i][m]) = 0  # Assign all possible upgrading heights

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cost function including annualized construction cost and expected annual damage cost
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Basic calcuation for leveed channel
Hfp = (W - B) * FS  # Floodplain height
Htoe = D + Hfp  # Water level at the toe of the levee


# Flow corresponding to aNY specific water level (from bottom of the river), calculated by Manning's Equation
def flow(waterh):
	if waterh >= Htoe:  # If water level is above the toe and below the top of the levee
		CrossSection = B * D + (B + W) * Hfp / 2 + (W + W + 2 * (waterh - Htoe) / WS) * (waterh - Htoe) / 2
		# Cross section area of flow at waterh depth
		WettedP = B + 2 * D + 2 * math.sqrt(((W - B) / 2) ** 2 + (Hfp) ** 2) + 2 * math.sqrt(
			((waterh - Htoe) / WS) ** 2 + (waterh - Htoe) ** 2)
		# Wetted perimeter
	else:
		if waterh >= D:  # If water level is above the channel depth and below the toe of the levee
			CrossSection = B * D + (B + B + 2 * (waterh - D) / FS) * (waterh - D) / 2
			# Cross section area of flow at waterh depth
			WettedP = B + 2 * D + 2 * math.sqrt(((waterh - D) / FS) ** 2 + (waterh - D) ** 2)
			# Wetted perimeter
		else:
			CrossSection = waterh * B  # Cross section area of flow at waterh depth
			WettedP = 2 * waterh + B  # Wetted perimeter
	Velocity = kconversion / Nc * (CrossSection / WettedP) ** (2 / 3) * math.sqrt(Sc)  # Water velocity
	overflow = Velocity * CrossSection  # Flow
	return overflow


QCMIN = flow(Htoe)  # The base channel capacity at the toe of the levee, below which there's no risk


# QCMIN = Qfp  The same as above, flow within the top of the floodplain

##################  Non-overflow intermediate failure probability
def integrand(Q, intemu, intesigma, inteheight):
	Htop = Htoe + inteheight
	QCMAX = flow(Htop)
	inteexpmu = math.exp(intemu)
	PQ = lognorm.pdf(Q, intesigma, loc=intemu, scale=inteexpmu)  # # Probability of Flow Q (PDF)
	if QCMAX == QCMIN:
		PNOVER = 1
	else:
		PNOVER = (Q - QCMIN) / (QCMAX - QCMIN)
	return PQ * PNOVER


def intermediatefailure(intemu, intesigma, inteheight):
	Htop = Htoe + inteheight
	QCMAX = flow(Htop)
	# FINTERMEDIATE = quad(integrand, QCMIN, QCMAX, args=(intemu, intesigma, inteheight))[0]
	inteq = np.arange(QCMIN, QCMAX + deltaq, deltaq)
	FINTERMEDIATE = simps(integrand(inteq, intemu, intesigma, inteheight), inteq)
	return FINTERMEDIATE


##################  Overflow failure probability
def overtopfailure(overmu, oversigma, overheight):
	Htop = Htoe + overheight
	OVERFLOW = flow(Htop)  # Overtopping Flow
	overexpmu = math.exp(overmu)
	FNOVER = lognorm.cdf(OVERFLOW, oversigma, loc=overmu, scale=overexpmu)
	FOVER = 1 - FNOVER
	return FOVER


################  Annually construction or upgrading costs of a levee, based on existing height and upgrading height
def constructioncost(conexisth, conupgradh):
	ALEVEE0 = (Bc + Bc + conexisth * (1 / WS + 1 / LS)) * conexisth / 2
	# Cross section area of the existing levee (m**2)
	ALEVEE1 = (Bc + Bc + (conupgradh + conexisth) * (1 / WS + 1 / LS)) * (conupgradh + conexisth) / 2
	# Cross section area of the upgraded levee (m**2)
	ALEVEE = ALEVEE1 - ALEVEE0
	# Upgrading cross section area of one levee (m**2)
	VOLUMN = ALEVEE * L  # Total volume of one levee (m^3)
	LC0 = Cland * L * (Bc + (1 / WS + 1 / LS) * conexisth)
	# Land use cost of the existing levee primarily for purchasing land ($)
	LC1 = Cland * L * (Bc + (1 / WS + 1 / LS) * (conupgradh + conexisth))
	# Land use cost of the upgraded levee primarily for purchasing land ($)
	LC = LC1 - LC0
	# Upgrading land use cost of one levee ($)
	cc = (Cadjust * Csoil * VOLUMN) + LC  # levee construction cost
	return cc


def annualtotalcost(EXISTH, UPGRADH, LOGMU, LOGSIGMA):
	cc = constructioncost(EXISTH, UPGRADH)
	CC = 2 * cc / 1000000.0  # construction cost of two levees in million $
	HEIGHT = EXISTH + UPGRADH
	FINTERMEDIATE = intermediatefailure(LOGMU, LOGSIGMA, HEIGHT)
	FOVER = overtopfailure(LOGMU, LOGSIGMA, HEIGHT)
	EAD = DC * (FINTERMEDIATE + FOVER)  # expected annual damage cost in million $
	# Expected annual total cost
	EAT = CC + EAD
	return EAT


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SDP Program with Markov Process
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 1 Calculation of all conditional probabilities for possible observed climate scenarios at previous stage

print str(
	time.ctime()) + " Calculation of all conditional probabilities for possible observed climate scenarios at previous stage "

PMUA = np.zeros(((n, NA, NA)))
# initialization of the conditional probability of a mean flow MU given a climate scenario, at time i=1:n
# (PMUA[i][k][s]) i =1:n stage or time; k =1:NA a mean flow MU; s =1:NA a climate scenario
for i in range(n):
	# stage or time i
	for k in range(NA):
		# the climate scenario or MU
		s = 0
		# a climate scenario
		PMUA0 = (norm.pdf((LOGMU[i, s]), (LOGMU[i][k]), (LOGSIGMA[i][k]))) ** 2
		# PMUA0 = norm.pdf( (LOGMU[i,s]), (LOGMU[i][k]), (LOGSIGMA[i][k]) )
		while s < NA - 1:
			s = s + 1
			PMUA0 = PMUA0 + (norm.pdf((LOGMU[i, s]), (LOGMU[i][k]), (LOGSIGMA[i][k]))) ** 2
			# PMUA0 = PMUA0 + norm.pdf( (LOGMU[i,s]), (LOGMU[i][k]), (LOGSIGMA[i][k]) )
		for s in range(NA):
			# a climate scenario
			(PMUA[i][k][s]) = (norm.pdf((LOGMU[i, s]), (LOGMU[i][k]), (LOGSIGMA[i][k]))) ** 2 / PMUA0
		# (PMUA[i][k][s]) = norm.pdf( (LOGMU[i,s]), (LOGMU[i][k]), (LOGSIGMA[i][k]) ) / PMUA0
		# lognorm.pdf( QMU, (LOGSIGMA[i][s]), loc = (LOGMU[i][s]), scale = EXPMU )  is the probability of flow (LOGMU[i][k]) given mean and variation
		# print i+1, k+1, s+1, '(PMUA[i][k][s])', (PMUA[i][k][s])

PAMU = np.zeros(((n, NA, NA)))
# initialization of the conditional probability of a current climate scenario (k=1:NA) given an observed climate in a previous climate scenario (j=1:NA), at time i=1:n
# (PAMU[i][j][k]) i =1:n stage or time; j =1:NA an observed previous climate scenario; k =1:NA a current climate scenario

probability = np.zeros((n * NA * NA, 4))
# initialization of the chaning probability of climate condition, varying with observed climate scenarios
# columns will be year, observed climate in previous year, current climate, probability

# the (first) starting stage climate scenarios info.
for i in range(1):
	# stage or time i
	for j in range(NA):
		# an observed previous climate scenario
		for k in range(NA):
			# a current climate scenario
			(PAMU[i][j][k]) = (PA[k])
			(probability[i * NA * NA + j * NA + k][0]) = i + 1
			(probability[i * NA * NA + j * NA + k][1]) = j + 1
			(probability[i * NA * NA + j * NA + k][2]) = k + 1
			(probability[i * NA * NA + j * NA + k][3]) = (PAMU[i][j][k])

# the rest stages climate scenarios info.
for i in range(1, n):
	# stage or time i
	for j in range(NA):
		# an observed previous climate scenario
		for k in range(NA):
			# a current climate scenario
			s = 0
			# a cliamte scenario
			PPMU = (PMUA[i - 1][j][s]) * (PA[s])
			while s < NA - 1:
				s = s + 1
				PPMU = PPMU + (PMUA[i - 1][j][s]) * (PA[s])
				# calculation for the probability of a previous observed climate scenario, PPMU
			(PAMU[i][j][k]) = (PA[k]) * (PMUA[i - 1][j][k]) / PPMU
			# the conditional probability of a climate scenario given an observed climate scenario
			# print i+1, j+1, k+1, '(PAMU[i][j][k])', (PAMU[i][j][k])
			(probability[i * NA * NA + j * NA + k][0]) = i + 1
			(probability[i * NA * NA + j * NA + k][1]) = j + 1
			(probability[i * NA * NA + j * NA + k][2]) = k + 1
			(probability[i * NA * NA + j * NA + k][3]) = (PAMU[i][j][k])

with open("SDPMP_probability_" + str(mus) + "_" + str(sigmas) + ".csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(probability)

print str(time.ctime()) + " documenting conditional probabilities "

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 2 Calculation of the backward SDP value functions

print str(time.ctime()) + " Calculation of the backward SDP value functions "

EATC = np.ones((((n, NA, NH, NHUP)))) * largenumber
# initialization of the expected annual total cost for upgrading height m=1:NHUP and existing height l=1:NH; given an observed previous climate scenario (j=1:NA), at time i=1:n 
# (EATC[i][j][l][m]) i =1:n stage or time; j =1:NA an observed previous climate scenario; l =1:NH existing levee height; m =1:NHUP upgrading levee height
VALUE = np.ones((((n, NA, NH, NHUP)))) * largenumber  # a very large number
# initialization of the accumulation for the decisions from all later stages, 
# for upgrading height m=1:NHUP and existing height l=1:NH, given an observed previous climate scenario (j=1:NA), at time i=1:n
# (VALUE[i][j][l][m]) i =1:n stage or time; j =1:NA an observed previous climate scenario; l =1:NH existing levee height; m =1:NHUP upgrading levee height
EMINIVALUE = np.ones((((n, NA, NH, NHUP)))) * largenumber  # a very large number
# initialization of the expection of the next stage minimum value for accumulation  
# given an observed previous climate scenario (j=1:NA), at time i=1:n, with current existing height l=1:NH and upgrading height m=1:NHUP
# (EMINIVALUE[i][j][l][m]) i =1:n stage or time; j =1:NA an observed previous climate scenario; l =1:NH existing levee height; m =1:NHUP upgrading levee height

# backward recursive function for the last stage
for i in range(n - 1, n):
	# stage or time i, backward index(n-1-i), starting from the last stage index(n-1)
	for j in range(NA):
		# an observed previous climate scenario
		for l in range(NH):
			# existing levee height l
			existh = (EXH[i][l])
			if (i % T) == 0:
				NHUPm = int(round((Hupmax) / DELTAH + 1))  # upgrading time periods, calculate for all possible upgrading heights (UPH[i][m]) = 0
			else:
				NHUPm = 1  # none upgrading time periods, upgradh = 0, so could only calculate for the first (UPH[i][1]) = 0
			for m in range(NHUPm):
				# upgrading levee height m
				upgradh = (UPH[i][m])
				# corresponding to (EXH[i+1][]) = (EXH[i][l])+(UPH[i][m])
				k = 0
				# current climate scenario
				logmu = (LOGMU[i][k])
				logsigma = (LOGSIGMA[i][k])
				if existh + upgradh <= Hmax + 1e-6:
					# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
					(EATC[i][j][l][m]) = annualtotalcost(existh, upgradh, logmu, logsigma) * (PAMU[i][j][k])
				while k < NA - 1:
					k = k + 1
					logmu = (LOGMU[i][k])
					logsigma = (LOGSIGMA[i][k])
					if existh + upgradh <= Hmax + 1e-6:
						(EATC[i][j][l][m]) = (EATC[i][j][l][m]) + annualtotalcost(existh, upgradh, logmu, logsigma) * (
						PAMU[i][j][k])
						# adding up for expected annual total cost over all possible climate scenarios at stage i
				if m == 0:
					LAC = 0
					# if there's no levee system upgrade, levee alteration cost is 0
				else:
					LAC = FLAC
					# if there's levee system upgrade at any height, levee alteration cost is the fixed cost
				(VALUE[i][j][l][m]) = (EATC[i][j][l][m] + AOM + LAC) * ((math.exp(R)) / (math.exp(R) - 1))

			# backward recursive function for the intermediate stages
for i in range(n - 2, 0, -1):
	# stage or time i, backward
	for j in range(NA):
		# an observed previous climate scenario
		for l in range(NH):
			# index for existing levee height l as state
			existh = (EXH[i][l])
			# existing levee height l as state
			if (i % T) == 0:
				NHUPm = int(round((Hupmax) / DELTAH + 1))
				# upgrading time periods, calculate for all possible upgrading heights (UPH[i][m]) = 0
			else:
				NHUPm = 1
				# none upgrading time periods, upgradh = 0, so could only calculate for the first (UPH[i][1]) = 0
			for m in range(NHUPm):
				# index for upgrading levee height m as decision variable
				upgradh = (UPH[i][m])
				# upgrading levee height m as decision variable, corresponding to existing levee height next stage (EXH[i][l])+(UPH[i][m])
				k = 0
				# current climate scenario
				MINIVALUE = largenumber
				# initialization of the next stage minimum value for accumulation
				logmu = (LOGMU[i][k])
				logsigma = (LOGSIGMA[i][k])
				# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
				if existh + upgradh <= Hmax + 1e-6:
					# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
					(EATC[i][j][l][m]) = annualtotalcost(existh, upgradh, logmu, logsigma) * (PAMU[i][j][k])
					for ll in range(NH):
						# next stage index for existing levee height
						if abs((EXH[i + 1][ll]) - existh - upgradh) < 1e-6:
							# the existing levee height next stage is (EXH[i][l])+(UPH[i][m])
							for mm in range(NHUP):
								# next stage index for upgrading levee height
								if (VALUE[i + 1][k][ll][mm]) < MINIVALUE:
									MINIVALUE = (VALUE[i + 1][k][ll][mm])
				(EMINIVALUE[i][j][l][m]) = MINIVALUE * (PAMU[i][j][k])
				while k < NA - 1:
					k = k + 1
					MINIVALUE = largenumber
					# initialization of the next stage minimum value for accumulation
					logmu = (LOGMU[i][k])
					logsigma = (LOGSIGMA[i][k])
					# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
					if existh + upgradh <= Hmax + 1e-6:
						# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
						(EATC[i][j][l][m]) = (EATC[i][j][l][m]) + annualtotalcost(existh, upgradh, logmu, logsigma) * (
						PAMU[i][j][k])
						# adding up for expected annual total cost over all possible climate scenarios at stage i
						for ll in range(NH):
							# next stage index for existing levee height
							if abs((EXH[i + 1][ll]) - existh - upgradh) < 1e-6:
								# the existing levee height next stage is (EXH[i][l])+(UPH[i][m])
								for mm in range(NHUP):
									# next stage index for upgrading levee height
									if (VALUE[i + 1][k][ll][mm]) < MINIVALUE:
										MINIVALUE = (VALUE[i + 1][k][ll][mm])
					(EMINIVALUE[i][j][l][m]) = (EMINIVALUE[i][j][l][m]) + MINIVALUE * (PAMU[i][j][k])
					# adding up for expected minimum future accumulated value at next stage i+1 over all possible current climate scenarios
				if m == 0:
					LAC = 0
					# if there's no levee system upgrade, levee alteration cost is 0
				else:
					LAC = FLAC
					# if there's levee system upgrade at any height, levee alteration cost is the fixed cost
				(VALUE[i][j][l][m]) = (EATC[i][j][l][m]) + AOM + LAC + (EMINIVALUE[i][j][l][m]) * (math.exp(-R * 1))

# backward recursive function for the (first) starting stage
for i in range(1):
	# stage or time i, backward
	for j in range(NA):
		# an observed previous climate scenario
		for l in range(1):
			# index for existing levee height l as state
			existh = (EXH[i][l])
			# existing levee height l as state
			for m in range(NHUP):
				# index for upgrading levee height m as decision variable
				upgradh = (UPH[i][m])
				# upgrading levee height m as decision variable, corresponding to existing levee height next stage (EXH[i][l])+(UPH[i][m])
				k = 0
				# current climate scenario
				MINIVALUE = largenumber
				# initialization of the next stage minimum value for accumulation
				logmu = (LOGMU[i][k])
				logsigma = (LOGSIGMA[i][k])
				# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
				if existh + upgradh <= Hmax + 1e-6:
					# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
					(EATC[i][j][l][m]) = annualtotalcost(existh, upgradh, logmu, logsigma) * (PAMU[i][j][k])
					for ll in range(NH):
						# next stage index for existing levee height
						if abs((EXH[i + 1][ll]) - existh - upgradh) < 1e-6:
							# the existing levee height next stage is (EXH[i][l])+(UPH[i][m])
							for mm in range(NHUP):
								# next stage index for upgrading levee height
								if (VALUE[i + 1][k][ll][mm]) < MINIVALUE:
									MINIVALUE = (VALUE[i + 1][k][ll][mm])
				(EMINIVALUE[i][j][l][m]) = MINIVALUE * (PAMU[i][j][k])
				while k < NA - 1:
					k = k + 1
					MINIVALUE = largenumber
					# initialization of the next stage minimum value for accumulation
					logmu = (LOGMU[i][k])
					logsigma = (LOGSIGMA[i][k])
					# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
					if existh + upgradh <= Hmax + 1e-6:
						# expected annual total cost only exists when existing + upgrading height is no greater than maximum height
						(EATC[i][j][l][m]) = (EATC[i][j][l][m]) + annualtotalcost(existh, upgradh, logmu, logsigma) * (
						PAMU[i][j][k])
						# adding up for expected annual total cost over all possible climate scenarios at stage i
						for ll in range(NH):
							# next stage index for existing levee height
							if abs((EXH[i + 1][ll]) - existh - upgradh) < 1e-6:
								# the existing levee height next stage is (EXH[i][l])+(UPH[i][m])
								for mm in range(NHUP):
									# next stage index for upgrading levee height
									if (VALUE[i + 1][k][ll][mm]) < MINIVALUE:
										MINIVALUE = (VALUE[i + 1][k][ll][mm])
					(EMINIVALUE[i][j][l][m]) = (EMINIVALUE[i][j][l][m]) + MINIVALUE * (PAMU[i][j][k])
					# adding up for expected minimum future accumulated value at next stage i+1 over all possible current climate scenarios
				if m == 0:
					LAC = 0
					# if there's no levee system upgrade, levee alteration cost is 0
				else:
					LAC = FLAC
					# if there's levee system upgrade at any height, levee alteration cost is the fixed cost
				(VALUE[i][j][l][m]) = (EATC[i][j][l][m]) + AOM + LAC + (EMINIVALUE[i][j][l][m]) * (math.exp(-R * 1))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialization for STEP3

print str(time.ctime()) + " Identifying the optimal solutions "

OPTVALUE = np.ones(((n, NH, NA))) * largenumber  # a very large number
# initialization of the accumulation for the best decisions from all later stages, at time i=1:n, for existing height l=1:NH, given an observed previous climate scenario (j=1:NA)
# (OPTVALUE[i][l][j]) i =1:n stage or time; l =1:NH existing levee height; j =1:NA an observed previous climate scenario
OPTUP = np.zeros(((n, NH, NA)))
# initialization of minimum upgrading height m, at time i=1:n, for existing height l=1:NH, given an observed previous climate scenario (j=1:NA)
# (OPTUP[i][l][j]) i =1:n stage or time; l =1:NH existing levee height; j =1:NA an observed previous climate scenario
OPTH = np.zeros(((n, NH, NA)))
# initialization of minimum resulted height l, at time i=1:n, for existing height l=1:NH, given an observed previous climate scenario (j=1:NA)
# (OPTH[i][l][j]) i =1:n stage or time; l =1:NH existing levee height; j =1:NA an observed previous climate scenario


# Initialization for outcome Post-processing
OPTRESULTS = np.zeros((NT * NH, 3 * NA + 2))  # Report only stages that can upgrade
# OPTRESULTS=np.zeros((n*NH,3*NA+2)) # Report all stages
# Rows are organized as subsequent stages from 1 to n, with every possible existing levee height NH in each stage
# Columns are organized as the index for stage (column 1), index for existing levee height (column 2), 
# OPTUP for each climate scenario (column 3~NA+2), OPTH for each climate scenario NA (column NA+3~2*NA+2), and OPTVALUE for each climate scenario (column 2*NA+3~3*NA+2)


for i in range(NT * NH):  # Report only stages that can upgrade
	# for i in range(n*NH): # Report all stages
	Rn = i / NH * T + 1  # Report only stages that can upgrade
	# Rn = i/NH + 1 # Report all stages
	# index for stage
	RNH = (i % NH) * DELTAH
	# index for existing levee height at each stage
	for j in range(1):
		(OPTRESULTS[i][j]) = Rn
	for j in range(1, 2):
		(OPTRESULTS[i][j]) = RNH
	for j in range(2 * NA + 2, 3 * NA + 2):
		(OPTRESULTS[i][j]) = largenumber

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 3 Identifying the optimal solutions

# Identifying the optimal results, starting from the first stage
for i in range(n):
	# stage or time i
	ii = i / T  # reporting only the upgrade stage
	for l in range(NH):
		# existing levee height l, starting from H0 = 0
		for j in range(NA):
			# an observed previous climate scenario, none for the first stage
			for m in range(NHUP):
				# upgrading levee height m
				if (VALUE[i][j][l][m]) < (OPTVALUE[i][l][j]):
					(OPTVALUE[i][l][j]) = (VALUE[i][j][l][m])
					(OPTUP[i][l][j]) = (UPH[i][m])
					(OPTH[i][l][j]) = (EXH[i][l]) + (OPTUP[i][l][j])
			# reporting only upgrade stage
			if (i % T) == 0:
				(OPTRESULTS[ii * NH + l][j + 2]) = (OPTUP[i][l][j])
				(OPTRESULTS[ii * NH + l][j + NA + 2]) = (OPTH[i][l][j])
				(OPTRESULTS[ii * NH + l][j + 2 * NA + 2]) = (OPTVALUE[i][l][j])
			# Reporting all
			# (OPTRESULTS[i*NH+l][j+2]) = (OPTUP[i][l][j])
			# (OPTRESULTS[i*NH+l][j+NA+2]) = (OPTH[i][l][j])
			# (OPTRESULTS[i*NH+l][j+2*NA+2]) = (OPTVALUE[i][l][j])
			if i == 0:
				if l == 0:
					print 'Stage:', i + 1, 'Exist H:', (EXH[i][l]), 'Ob Climate:', j + 1, 'Opt Up H:', (
					OPTUP[i][l][j]), 'Result H:', (OPTH[i][l][j]), 'Opt Value:', (OPTVALUE[i][l][j])
					# if ( i % T ) == 0:
				# print 'Stage:', i+1, 'Exist H:', (EXH[i][l]), 'Ob Climate:', j+1, 'Opt Up H:', (OPTUP[i][l][j]), 'Result H:', (OPTH[i][l][j]), 'Opt Value:', (OPTVALUE[i][l][j])

with open("SDPMP_" + str(T) + "_" + str(mus) + "_" + str(sigmas) + ".csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(OPTRESULTS)

# NUMBERS in csv file name represent year interval, number of interval and number of climate scenarios  
print str(time.ctime()) + " documenting optimal results"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Reorganizing the optimal planning pathways, starting from the first stage

OPT_RESULTS = np.zeros((1, 3 * NA + 2))  # Initialization of optimal planning

for i in range(1):
	(OPT_RESULTS[i, :]) = (OPTRESULTS[i, :])
	opthlist = [(OPT_RESULTS[i, NA + 2:2 * NA + 2])]
	opthnewlist = np.unique(opthlist)
	# print OPT_RESULTS, opthlist, opthnewlist

for i in range(1, n):  # stage
	ii = i / T
	rowexist = OPT_RESULTS.shape[0]  # number of rows of the previous matrix OPT_RESULTS
	# print 'rowexist', rowexist
	if (i % T) == 0:  # upgrade stage
		for l in range(70, NH):  # any existing height
			for x in opthnewlist:  # a optimal solution
				if abs((OPTRESULTS[ii * NH + l, 1]) - x) < 1e-6:
					addrow = (OPTRESULTS[ii * NH + l, :])
					OPT_RESULTS = np.vstack([OPT_RESULTS, addrow])
					# print l, j, (OPTRESULTS[ii*NH+l,1]), 'addrow', addrow
		rownew = OPT_RESULTS.shape[0]  # number of rows of the current matrix OPT_RESULTS
		k = rownew - 1
		if abs((OPT_RESULTS[k, 0]) - (i + 1)) < 1e-6:  # for any possibl｀e optimal path of current stage
			opthlist = [(OPT_RESULTS[k, NA + 2:2 * NA + 2])]
		while k > 0:
			k = k - 1
			if abs((OPT_RESULTS[k, 0]) - (i + 1)) < 1e-6:  # for any possibl｀e optimal path of current stage
				opthlist = np.vstack([opthlist, (OPT_RESULTS[k, NA + 2:2 * NA + 2])])
		opthnewlist = np.unique(opthlist)
		# print OPT_RESULTS, opthlist, opthnewlist
	else:
		rowexist = OPT_RESULTS.shape[0]  # number of rows of the previous matrix OPT_RESULTS
		for x in opthnewlist:  # a optimal solution
			addrow = (OPT_RESULTS[rowexist - 1, :])
			OPT_RESULTS = np.vstack([OPT_RESULTS, addrow])
			# print 'addrow', addrow, 'OPT_RESULTS', OPT_RESULTS,
			rowup = OPT_RESULTS.shape[0]  # the added stage number to be updated
			OPT_RESULTS[rowup - 1, 0] = i + 1
			OPT_RESULTS[rowup - 1, 1] = x
			for jj in range(NA):
				OPT_RESULTS[rowup - 1, jj + 2] = 0
				OPT_RESULTS[rowup - 1, jj + NA + 2] = x
			# print 'rowup', rowup, 'OPT_RESULTS', OPT_RESULTS
		rownew = OPT_RESULTS.shape[0]  # number of rows of the current matrix OPT_RESULTS
		k = rownew - 1
		if abs((OPT_RESULTS[k, 0]) - (i + 1)) < 1e-6:  # for any possible optimal path of current stage
			opthlist = [(OPT_RESULTS[k, NA + 2:2 * NA + 2])]
		while k > 0:
			k = k - 1
			if abs((OPT_RESULTS[k, 0]) - (i + 1)) < 1e-6:  # for any possible optimal path of current stage
				opthlist = np.vstack([opthlist, (OPT_RESULTS[k, NA + 2:2 * NA + 2])])
		opthnewlist = np.unique(opthlist)
		# print OPT_RESULTS, opthlist, opthnewlist

rowoptpath = OPT_RESULTS.shape[0]  # number of rows of the optimal results path matrix OPT_RESULTS
for k in range(rowoptpath):
	if (OPT_RESULTS[k, 0]) == 200:
		print (OPT_RESULTS[k, 8:13])

with open("SDPMP_PATH_" + str(T) + "_" + str(mus) + "_" + str(sigmas) + ".csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(OPT_RESULTS)

print str(time.ctime()) + "documenting optimal planning pathy for plot"

print "Computation time", (time.time() - start)

print 'Figure 1'
plt.figure(1)
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA], 'ro', label='climate 1')
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 1], 'bs', label='climate 2')
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 2], 'yd', label='climate 3')
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 3], 'k*', label='climate 4')
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 4], 'g^', label='climate 5')
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 5], 'cp', label='climate 6')
# plt.axis([0, 200, 7.5, 10])
plt.xlabel('planning stage(year)', fontsize=16)
plt.ylabel('levee height (m)', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.show()

print 'Figure 2'
plt.figure(2)
plt.subplot(231)
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA], 'ro', label='climate 1')
# plt.axis([0, 200, 7.5, 10])
plt.xlabel('planning stage(year)', fontsize=16)
plt.ylabel('levee height (m)', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.subplot(232)
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 1], 'bs', label='climate 2')
# plt.axis([0, 200, 7.5, 10])
plt.xlabel('planning stage(year)', fontsize=16)
plt.ylabel('levee height (m)', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.subplot(233)
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 2], 'yd', label='climate 3')
# plt.axis([0, 200, 7.5, 10])
plt.xlabel('planning stage(year)', fontsize=16)
plt.ylabel('levee height (m)', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.subplot(234)
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 3], 'k*', label='climate 4')
# plt.axis([0, 200, 7.5, 10])
plt.xlabel('planning stage(year)', fontsize=16)
plt.ylabel('levee height (m)', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.subplot(235)
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 4], 'g^', label='climate 5')
# plt.axis([0, 200, 7.5, 10])
plt.xlabel('planning stage(year)', fontsize=16)
plt.ylabel('levee height (m)', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.subplot(236)
plt.plot(OPT_RESULTS[:, 0], OPT_RESULTS[:, 2 + NA + 5], 'cp', label='climate 6')
# plt.axis([0, 200, 7.5, 10])
plt.xlabel('planning stage(year)', fontsize=16)
plt.ylabel('levee height (m)', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


print str(time.ctime()) + " Done!"

end = time.time()
print "Program run time", (end - start)

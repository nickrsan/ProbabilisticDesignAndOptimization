import random
import math
import seaborn
import numpy

import matplotlib.pyplot as plt

A = 500
R = 30
S = 0.01

random.seed(20190204)

seaborn.set(color_codes=True)


def manning(a=A, r=R, s=S, max_n=0.08, min_n=0.06):
	n_base = random.random()  # number between 0 and 1
	n_scaled = n_base * (max_n - min_n)  # scale it so it falls in the size range of min-max
	n = n_scaled + min_n  # then boost it by min_n so that size aligns with the actual values

	return (1.49/n)*a*(r**(2/3))*math.sqrt(s)

qs = []
for _ in range(100000):
	qs.append(manning())

seaborn.distplot(numpy.array(qs), bins=10)
plt.show()



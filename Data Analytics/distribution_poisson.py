#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:15:26 2020

@author: joke
"""

from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sb

fig, ax = plt.subplots(1, 1)

p = 0.02
n = 100

mu = (p*n)

x = 0
prob = poisson.pmf(x, mu)

print(prob)

ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)

ax.legend(loc='best', frameon=False)
plt.show()

data_rdm = poisson.rvs(mu=2, size=1000)

ax = sb.distplot(data_rdm,
                  kde=True,
                  color='green',
                  hist_kws={"linewidth": 25,'alpha':1})
ax.set(xlabel='Poisson', ylabel='Frequency')
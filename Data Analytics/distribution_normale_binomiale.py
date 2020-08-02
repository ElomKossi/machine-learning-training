#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:00:26 2020

@author: joke
"""

import matplotlib.pyplot as plt
import numpy as np

#Distribution normale

#X= Z* σ+ μ

mu = 0.6
sigma = 0.2

dtst = np.random.normal(mu, sigma, 10000)

count, bins, ignored = plt.hist(dtst, bins=20, normed=1,color="lightblue")

plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    np.exp( - (bins - mu)**2 / (2 * sigma**2) ),       linewidth=3, color='r')


plt.show()

#Distribution binomiale 

n=10
p = 0.5

data_binom = np.random.binomial(n, p, 1000)

answer=sum(np.random.binomial(9, 0.1, 20000) == 0)/20000
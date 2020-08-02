#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:25:41 2020

@author: joke
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import stats

dtst = pd.read_csv('iris.csv')

#sans regression
plot = sns.pairplot(dtst, kind="scatter")
plt.show()
plot.savefig('dispersion3.pdf', format='pdf')

X = dtst['longueur_petal']
Y = dtst['largeur_petal']
cor = stats.pearsonr(X, Y)
print(cor)

Y1 = dtst['longueur_sepal']
X1 = dtst['largeur_sepal']
noncor = stats.pearsonr(X1, Y1)
print(noncor)
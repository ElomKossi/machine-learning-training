#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:39:40 2020

@author: joke
"""
"""
#Nuage de points

import pandas as pd
import numpy as np

dataset = pd.DataFrame(np.random.rand(25, 2), columns=['ventes', 'benefice'])

dataset.plot.scatter(x='ventes', y='benefice')
"""

#============================================================

"""
#Graphique en 3D

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

chart = plt.figure()
chart3d = chart.add_subplot(111,projection='3d')

axe_X, axe_Y, axe_Z = axes3d.get_test_data(0.10)

chart3d.plot_wireframe(axe_X, axe_Y, axe_Z,color='r',rstride=15,cstride=10)

plt.show()
plt.style.use(['dark_background', 'fast'])
"""

#============================================================

from pylab import *
from mpl_toolkits.mplot3d import Axes3D


fig = figure()
ax = Axes3D(fig)

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

show()
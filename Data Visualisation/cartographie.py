#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:23:28 2020

@author: joke
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs  

fig = plt.figure(figsize=(25, 15))
axe = fig.add_subplot(1, 1, 1, projection=ccrs.InterruptedGoodeHomolosine())

"""
Liste de projection Cartopy
    AzimuthalEquidistant : Une projection azimutale équidistante
    LambertCylindrical 
    Mollweide
    Robinson
    InterruptedGoodeHomolosine
"""

axe.set_extent((251, 21, 75, -35))

axe.stock_img()
axe.coastlines()

axe.tissot(facecolor='purple', alpha=0.8)
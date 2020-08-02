#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:37:14 2020

@author: joke
"""

import pandas as pd

dataset = pd.read_csv("tendance_centrale.csv")

print ("\nLe minimum ")
print (dataset.min())
print ("\nLe maxiimum ")
print (dataset.max())

#La moyenne
print ("La moyenne ")
print (dataset.mean())

#La médianne
print ("\nLa médianne ")
print (dataset.median())

#Le mode
print ("\nLe mode de mesure ")
print (dataset.mode())

#L'équart type
print ("\nL'équart type de mesure ")
print (dataset.std())

#L'équart type
print ("\nL'asymétrie ")
print (dataset.skew())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:26:59 2020

@author: joke
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dtst = pd.read_csv('style.csv')

axe_abscisse = np.arange(0,20) 

droit_c = dtst.iloc[:,0].values
droit_d = dtst.iloc[:,1].values
droit_e = dtst.iloc[:,-1].values

#Labeliser l'Axes et Titre
plt.title("évolution du portefeuille") 
plt.xlabel("Temps") 
plt.ylabel("Montant du capitale") 

plt.plot(axe_abscisse,droit_c)
plt.plot(axe_abscisse,droit_d)
plt.plot(axe_abscisse,droit_e)

plt.annotate(xy=[2,1], s='faible Evolution du chiffre_affaires')
plt.annotate(xy=[4,6], s='forte croissance du bénéfice') 
plt.annotate(xy=[12,15], s='forte croissance du PIB') 

plt.legend(['Legende 1', 'Legende 2','Legende 3'], loc=4)

# Style de l'arriere plan
plt.style.use(['dark_background','fast'])

# sauvegarde en format pdf
plt.savefig('style.pdf', format='pdf')
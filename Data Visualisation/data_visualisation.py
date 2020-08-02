#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:45:34 2020

@author: joke
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dtst = pd.read_csv('graph_1.csv')
axe_ordonnee = dtst.iloc[:,0]
axe_abscisse = dtst.iloc[:,-1]

#Mise  en place de place de graphe simple
plt.plot(axe_abscisse,axe_ordonnee)

#Labeliser l'Axes et Titre
plt.title("Evolution du portefeuille")
plt.xlabel("Temps")
plt.ylabel("Montant du capitale") 

# Formater la lgne de couleur
plt.plot(axe_abscisse,axe_ordonnee,'green')

# sauvegarde en format pdf 
plt.savefig('illustration_graph_1.pdf', format='pdf')
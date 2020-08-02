#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 22:44:21 2020

@author: joke
"""

import pandas as pd
import numpy as np

dataset = pd.DataFrame(np.random.randn(6,4),index=['1','3','4','6','7','8'], 
                       columns=['taux_de_vente','croissance_vente','ratio_benefice','ratio_perte'])
print(dataset, '\n')

dataset = dataset.reindex(['1','2','3','4','5','6','7','8'])
print(dataset, '\n')

#afficher les null en true
print (dataset.isnull(), '\n')

#afficher les null en true
dataset = dataset[dataset.isnull().any(axis=1)]
print(dataset, '\n')


#Remplacer NanN par une valeur scalaire
print (dataset.fillna(0), '\n')

#Supprimer les valeurs manquants
print (dataset.dropna(), '\n')
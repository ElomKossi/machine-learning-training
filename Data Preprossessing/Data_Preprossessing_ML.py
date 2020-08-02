# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 21:45:27 2019

@author: joke
"""

import statsmodels as stat
import seaborn as sbrn
import pandas as pds
import matplotlib.pyplot as mplt
import numpy as np 
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize


dtst = pds.read_csv('credit_immo.csv')
print(dtst.shape) #Pour visualiser la taille des donner (19 sur 10)
print(dtst.describe()) # Pour décrire les données
print(dtst.columns)

X = dtst.iloc[:,-9:-1].values
Y = dtst.iloc[:,-1].values

print(dtst.isnull().sum)

imptr = Imputer(missing_values= 'NaN',strategy = 'mean',axis = 0)

imptr.fit(X[:,0:1])
imptr.fit(X[:,7:8])

X[:,0:1] = imptr.transform(X[:,0:1]) 
#Imputez toutes les valeurs manquantes dans X
X[:,7:8] = imptr.transform(X[:,7:8])

#Données catégoriques
labEncr_X = LabelEncoder()
X[:,2] = labEncr_X.fit_transform(X[:,2])
X[:,5] = labEncr_X.fit_transform(X[:,5])  
#OneHotEncoder = Encode les entités entières catégoriques sous la forme d'un tableau à une seule valeur.
onehotEncr = OneHotEncoder(categorical_features=[2])
onehotEncr = OneHotEncoder(categorical_features=[5])

X = onehotEncr.fit_transform(X).toarray()

#LabelEncoder	Encode les étiquettes avec une valeur comprise entre 0 et n_classes-1
labEncr_Y = LabelEncoder()
Y = labEncr_Y.fit_transform(Y)

# Fractionner l'ensemble de données en train et ensemble d'essai
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#standardisation (centrer-réduire )" signifie conversion vers un standard commun

StdSc = StandardScaler()

X_train = StdSc.fit_transform(X_train)
X_test = StdSc.fit_transform(X_test)


X_train = normalize(X_train)
X_test = normalize(X_test)





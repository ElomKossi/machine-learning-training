# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:41:22 2019

@author: joke
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Charger le jeu de données.
dataset = pd.read_csv('qualite-vin-rouge.csv')

# Qualité le paramètre à prédire est représenté par X.
y = dataset[['qualité']]
# Tous les paramètres d'entrée utilisés pour prédire la valeur sont représentés par y.
X = dataset[['acidité fixe','acidité volatile','acide citrique','sucre résiduel','chlorures','dioxyde de soufre libre','anhydride sulfureux total','densité','pH','sulphates','alcool']]

y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2)

model = PolynomialFeatures(degree= 4)
poly_features = model.fit_transform(X)
poly_features_test = model.fit_transform(X_test)


lg = LinearRegression()
lg.fit(poly_features,y)
donnee_pred = lg.predict(poly_features_test)

#Evaluer le model
print (mean_squared_error(y_test,donnee_pred))

###################################################
#model 2 degré 5
model_2 = PolynomialFeatures(degree= 3)
poly_features_2 = model_2.fit_transform(X)

poly_features_test_2 = model_2.fit_transform(X_test)



lg_2 = LinearRegression()
lg_2.fit(poly_features_2,y)
donnee_pred_2 = lg_2.predict(poly_features_test_2)

print (mean_squared_error(y_test,donnee_pred_2))






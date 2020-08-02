# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 23:10:26 2019

@author: joke
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


data = pd.read_csv("boston_house_prices.csv")
X = data.drop("MEDV", axis=1)
X = X[['CRIM','ZN','INDUS','NOX','RM','DIS','RAD','TAX','PTRATIO','B','LSTAT']].values
y = data.MEDV

# Fractionnement du dataset entre le Training set et le Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

regressor = LinearRegression()
#J'adapte le modèle de régression linéaire à l'ensemble de données d'apprentissage.
regressor.fit(X_train, y_train)

# Faire de nouvelles prédictions
y_pred = regressor.predict(X_test)
#================================EVALUATION MODEL 2
erreur_quadratique_moyenne = np.mean((regressor.predict(X_test) - y_test)**2)
print(erreur_quadratique_moyenne)


model1 = sm.OLS(y_train, X_train)
result= model1.fit( )
print(result.summary()) 








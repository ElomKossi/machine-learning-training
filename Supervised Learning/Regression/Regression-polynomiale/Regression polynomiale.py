# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:09:50 2019

@author: joke 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

## charger le jeu de données
dataset= pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, -1].values


# ajout de polynômes pour un meilleur ajustement des données
model = PolynomialFeatures(degree= 2)
X_poly_features = model.fit_transform(X)
model.fit(X,y)

poly_regression = LinearRegression()
poly_regression.fit(X_poly_features,y)

# régression normale
Linreg=LinearRegression()
Linreg.fit(X,y)


plt.scatter(X,y)
plt.plot(X,poly_regression.predict(X_poly_features))
plt.title("Régression polynomiale  Expérience par rapport au  Salaire 2")
plt.xlabel("Expérience ")
plt.ylabel("Salaire ")
plt.show()


plt.scatter(X,y)
plt.plot(X,Linreg.predict(X))
plt.title("Regression  Linaire  Expérience par rapport au  Salaire  ")
plt.xlabel("Expérience ")
plt.ylabel("Salaire ")
plt.show()



# Ajout de polynominaux à l'hypothèse
model = PolynomialFeatures(degree= 5)
X_poly_features = model.fit_transform(X)
model.fit(X,y)
poly_regression = LinearRegression()
poly_regression.fit(X_poly_features,y) 


plt.scatter(X,y)
plt.plot(X,poly_regression.predict(X_poly_features))
plt.title("Régression polynomiale  Expérience par rapport au  Salaire ")
plt.xlabel("Expérience ")
plt.ylabel("Salaire ")
plt.show()
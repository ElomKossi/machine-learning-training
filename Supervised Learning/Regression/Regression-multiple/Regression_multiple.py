# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:15:10 2019

@author: joke
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pltµ
from sklearn.linear_model import LinearRegression
import seaborn


data = pd.read_csv("boston_house_prices.csv")
X = data.drop("MEDV",axis=1)
y = data.MEDV


plt.figure(figsize=(75, 5))

for i, col in enumerate(X.columns):
    plt.subplot(1, 13, i+1)
    x = data[col]
    y = y
    plt.plot(x, y, 'o')
    # Crétation de la ligne de regression 
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.style.use(['dark_background','fast'])
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prix')
    
# Fractionnement du dataset entre le Training set et le Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalisation des données
scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Construction du modèle
regressor = LinearRegression()
#J'adapte le modèle de régression linéaire à l'ensemble de données d'apprentissage.
regressor.fit(X_train, y_train)


# Faire de nouvelles prédictions
y_pred = regressor.predict(X_test)

plt.style.use("bmh")
plt.scatter(y_pred,y_test)
plt.show




regressor.predict(scaler.fit_transform(np.array([[0.17331,0,9.69,0,0.585,5.707,54,2.3817,6,391,19.2,396.9,12.01]])))


#================Evaluation et validation ========================================================
# Le B0 de la fonction 
constante = regressor.intercept_
print(constante)

# Les variables explicative
coefficients = regressor.coef_
print(regressor.coef_)

nom =[i for i in list (X)]

erreur_quadratique_moyenne = np.mean((y_pred - y_test)**2)
print(erreur_quadratique_moyenne)

# L'évaluation
import statsmodels.api as sm

model1 = sm.OLS(y_train, X_train)

result= model1.fit( )
print(result.summary()) 


# Faire face à la multicolinéarité #############################

corr_df=X_train.corr(method='pearson')
print("--------Creer un graph  de  Correlation ")

mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True


seaborn.heatmap (corr_df, cmap='RdYlGn_r', vmax=1.0,vmin=-1.0,mask = mask, linewidths=2.5)

#affichage
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

import seaborn as sns
sns.pairplot(X_train, kind="scatter")
plt.show()


















